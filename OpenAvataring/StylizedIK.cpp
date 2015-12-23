#include "pch_bcl.h"
#include "StylizedIK.h"
#pragma warning (push)
#pragma warning (disable : 4297 )
#include <dlib\optimization\optimization.h>
#pragma warning (pop)
#include <unsupported\Eigen\LevenbergMarquardt>
#include "Settings.h"

using namespace Causality;
using namespace std;
using namespace Eigen;
using namespace DirectX;

typedef dlib::matrix<double, 0, 1> dlib_vector;

template <class DerivedX, class DerivedY>
dlib_vector ComposeOptimizeVector(const DenseBase<DerivedX> &x, const DenseBase<DerivedY> &y);

void DecomposeOptimizeVector(const dlib_vector& v, RowVectorXd &x, RowVectorXd &y);

template<class DerivedX, class DerivedY>
inline dlib_vector ComposeOptimizeVector(const DenseBase<DerivedX>& x, const DenseBase<DerivedY>& y)
{
	dlib_vector v(x.size() + y.size());
	RowVectorXd::Map(v.begin(), x.size()) = x;
	RowVectorXd::Map(v.begin() + x.size(), y.size()) = y;
	return v;
}

StylizedChainIK::StylizedChainIK()
{
	Reset();
}

StylizedChainIK::StylizedChainIK(const std::vector<const Joint*>& joints, ArmatureFrameConstView defaultframe)
	: StylizedChainIK()
{
	SetChain(joints, defaultframe);
}

void Causality::StylizedChainIK::SetFeatureDecoder(std::unique_ptr<IFeatureDecoder>&& decoder)
{
	m_fpDecoder = move(decoder);
}

void Causality::StylizedChainIK::SetChain(const std::vector<const Joint*>& joints, ArmatureFrameConstView defaultframe)
{
	m_chain.resize(joints.size());
	m_chainRot.resize(joints.size());
	//m_iy.setZero(joints.size() * 3);
	m_chainLength = 0;
	for (int i = 0; i < joints.size(); i++)
	{
		auto& bone = defaultframe[joints[i]->ID];
		m_chain[i].w = 1.0f;
		reinterpret_cast<DirectX::Vector3&>(m_chain[i]) = bone.LclTranslation;
		m_chainLength += bone.LclTranslation.Length();

		 Vector3f lq;
		 XMStoreFloat3(lq.data(),XMQuaternionLn(XMLoadA(bone.LclRotation)));
		 //m_iy.segment<3>(i * 3) = lq.cast<double>();
	}

	m_iy = m_gplvm.uY;
}

void Causality::StylizedChainIK::SetGoal(const Eigen::Vector3d & goal)
{
	m_goal = goal;
}

void StylizedChainIK::SetIKWeight(double weight)
{
	m_ikWeight = weight;
}

void StylizedChainIK::SetMarkovWeight(double weight)
{
	m_markovWeight = weight;
}

void Causality::StylizedChainIK::SetBaseRotation(const Quaternion & q)
{
	m_baseRot = q;
}

void Causality::StylizedChainIK::SetHint(const Eigen::RowVectorXd & y)
{
	//m_cy = y;
	m_iy = y;
	//m_cValiad = true;
}

DirectX::XMVECTOR StylizedChainIK::EndPosition(const XMFLOAT4A* rotqs)
{
	const auto n = m_chain.size();

	XMVECTOR q, t, gt, gq;
	Eigen::Vector4f qs;
	qs.setZero();

	gt = XMVectorZero();
	gq = XMLoad(m_baseRot);

	for (int i = 0; i < n; i++)
	{
		q = XMLoadFloat4A(&rotqs[i]);
		t = m_chain[i];

		t = XMVector3Rotate(t, gq);
		gq = XMQuaternionMultiply(q, gq);

		gt += t;
	}

	return gt;
}

void AbsoluteLnQuaternionDecode(_Out_cap_(n) DirectX::Quaternion* rots, const Eigen::RowVectorXd& y)
{
	int n = y.size() / 3;
	Eigen::Vector4f qs;
	XMVECTOR q;
	qs.setZero();
	for (int i = 0; i < n; i++)
	{
		qs.segment<3>(0) = y.segment<3>(i * 3).cast<float>();
		q = XMLoadFloat4A(qs.data());
		q = XMQuaternionExp(q); // revert the log map
		XMStoreA(rots[i], q);
	}

	//return rots;
}

Eigen::Matrix3Xf StylizedChainIK::EndPositionJacobi(const XMFLOAT4A* rotqs)
{
	const auto n = m_chain.size();

	Matrix3Xf jacb(3, 3 * n);
	MatrixXf rad(3, n);


	XMVECTOR q, t, gt, gq;

	gt = XMVectorZero();
	gq = XMQuaternionIdentity();
	t = XMVectorZero();

	for (int i = n-1; i >= 0; i--)
	{
		q = XMLoadFloat4A(&rotqs[i]);

		gt = XMVector3Rotate(gt, q);
		t = XMLoadA(m_chain[i]);
		XMStoreFloat3(rad.col(i).data(), gt);
		gt += t;
	}

	XMMATRIX rot;
	XMFLOAT4X4A jac;
	jac._11 = jac._22 = jac._33 = 0;
	jac._41 = jac._42 = jac._43 = jac._44 = jac._14 = jac._24 = jac._34 = 0;

	gq = XMLoad(m_baseRot);
	for (int i = 0; i < n; i++)
	{
		q = XMLoadFloat4A(&rotqs[i]);

		auto& r = rad.col(i);
		JacobbiFromR(jac, r.data());

		rot = XMMatrixRotationQuaternion(XMQuaternionConjugate(gq));
		// transpose as XMMatrix is row major
		rot = XMMatrixMultiplyTranspose(rot, XMLoadA(jac));
		for (int j = 0; j < 3; j++)
		{
			XMStoreFloat3(jacb.col(i*3+j).data(), rot.r[j]);
		}
		
		gq = XMQuaternionMultiply(q, gq);
	}

	return jacb;
}

void Causality::StylizedChainIK::JacobbiFromR(DirectX::XMFLOAT4X4A &jac, _In_reads_(3) const float* r)
{
	jac._12 = r[2];
	jac._13 = -r[1];
	jac._21 = -r[2];
	jac._23 = r[0];
	jac._31 = r[1];
	jac._32 = -r[0];
}

double StylizedChainIK::objective(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y)
{
	const auto n = m_chain.size();

	(*m_fpDecoder)(m_chainRot.data(), y);

	XMVECTOR ep = EndPosition(reinterpret_cast<XMFLOAT4A*>(m_chainRot.data()));
	Vector3f epf;
	XMStoreFloat3(epf.data(), ep);

	double ikdis = (epf.cast<double>() - m_goal).cwiseAbs2().sum() * m_ikWeight;

	//double iklimdis = ((m_limy.row(0) - y).cwiseMax(y - m_limy.row(1))).cwiseMax(0).cwiseAbs2().sum() * m_ikLimitWeight;

	double markovdis = (y - m_cy).cwiseAbs2().sum() * m_markovWeight;

	double fitlikelihood = (y.array() /** m_wy.array()*/ - m_ey.array()).cwiseAbs2().sum() * (0.5 * m_styleWeight / m_segmaX);
	//double fitlikelihood = m_gplvm.get_likelihood_xy(x, y) * g_StyleLikelihoodTermWeight;

	return ikdis + fitlikelihood + markovdis;//+iklimdis;
}

RowVectorXd StylizedChainIK::objective_derv(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y)
{
	const auto n = m_chain.size();

	RowVectorXd derv(x.size() + y.size());

	(*m_fpDecoder)(m_chainRot.data(), y);

	XMVECTOR ep = EndPosition(reinterpret_cast<XMFLOAT4A*>(m_chainRot.data()));
	MatrixXd jacb = EndPositionJacobi(reinterpret_cast<XMFLOAT4A*>(m_chainRot.data())).cast<double>();

	m_fpDecoder->EncodeJacobi(jacb);

	Vector3f epf;
	Vector3f goalf = m_goal.cast<float>();
	XMStoreFloat3(epf.data(), ep);

	// IK term derv
	RowVectorXd ikderv = (2.0 * m_ikWeight * (epf - goalf)).transpose().cast<double>() * jacb;

	//RowVectorXd iklimderv = 2.0 * m_ikLimitWeight * ((y - m_limy.row(0)).cwiseMin(0) + (y - m_limy.row(1)).cwiseMax(0));

	// Markov progation derv
	RowVectorXd markovderv = 2.0 * m_markovWeight * (y - m_cy);
	//markovderv.setZero();

	RowVectorXd animLkderv = (y.array()/* * m_wy.array()*/ - m_ey.array()) * (m_styleWeight / m_segmaX ); //m_gplvm.get_likelihood_xy_derivative(x, y) * g_StyleLikelihoodTermWeight;


	derv.segment(x.size(), y.size()) = ikderv + markovderv;// + gplvm.likelihood_xy_derv;
	derv.segment(0, x.size()).setZero();
	derv.segment(x.size(), y.size()) += animLkderv;
	//derv.segment(x.size(), y.size()) += iklimderv;

	return derv;//derv
}


// Filter alike interface
// reset history data

void StylizedChainIK::Reset()
{
	m_counter = 0;
	m_meanLk = 0;
	m_cValiad = false;
	m_cValiad = false;
	m_ikWeight = g_IKTermWeight;
	m_markovWeight = g_MarkovTermWeight;
	m_styleWeight = g_StyleLikelihoodTermWeight;
	m_ikLimitWeight = g_IKLimitWeight;
	m_baseRot = Quaternion::Identity;
	m_fpDecoder.reset(new AbsoluteLnQuaternionDecoder());
}

void DecomposeOptimizeVector(const dlib_vector & v, RowVectorXd & x, RowVectorXd & y)
{
	x = RowVectorXd::Map(v.begin(), x.size());
	y = RowVectorXd::Map(v.begin() + x.size(), y.size());
}

// return the joints rotation vector

const RowVectorXd & StylizedChainIK::Apply(const Vector3d & goal)
{
	if (!m_cValiad)
	{
		m_cx = goal;
		m_cy = m_iy;
		m_cValiad = true;
	}

	if (goal.norm() <= m_chainLength)
		m_goal = goal;
	else
		m_goal = goal * (m_chainLength / goal.norm());

	RowVectorXd hint_y;
	double lk = m_gplvm.get_expectation_and_likelihood(m_goal, &hint_y);
	lk = exp(-lk);

	m_meanLk = (m_meanLk * m_counter + lk) / (m_counter + 1);
	++m_counter;
	if (lk < 0.001 * m_meanLk) // low possibility prediction, use previous frame's data instead
		hint_y = m_cy;
	//hint_y = m_iy;

	m_cx = m_goal;
	return Apply(m_goal, hint_y.transpose().eval());
	
}

const Eigen::RowVectorXd & Causality::StylizedChainIK::Apply(const Eigen::Vector3d & goal, const const Eigen::Vector3d & goal_velocity)
{
	if (goal.norm() <= m_chainLength)
		m_goal = goal;
	else
		m_goal = goal * (m_chainLength / goal.norm());

	RowVectorXd x(1, 6);
	x.segment<3>(0) = goal;
	x.segment<3>(3) = goal_velocity;

	RowVectorXd hint_y;
	double lk = m_gplvm.get_expectation_and_likelihood(x, &hint_y);
	//hint_y.array() /= m_wy.array();
	lk = exp(-lk);

	if (!m_cValiad)
	{
		m_cx = goal;
		m_cy = hint_y;
		m_cValiad = true;
	}

	m_meanLk = (m_meanLk * m_counter + lk) / (m_counter + 1);
	++m_counter;

	//if (lk < 0.001 * m_meanLk) // low possibility prediction, use previous frame's data instead
		hint_y = m_cy;

	m_cx = x;

	return Apply(m_goal, hint_y.transpose().eval());//
}

const Eigen::RowVectorXd & Causality::StylizedChainIK::Apply(const Eigen::Vector3d & goal, const const Eigen::VectorXd & hint_y)
{

	auto v = ComposeOptimizeVector(m_cx, hint_y);

	//auto vmin = ComposeOptimizeVector(m_cx * 0.9, m_limy.row(0));
	//auto vmax = ComposeOptimizeVector(m_cx * 1.1, m_limy.row(1));

	m_segmaX = m_gplvm.get_expectation_and_likelihood(m_cx, &m_ey);
	m_segmaX = exp(2 / m_cy.cols() * m_segmaX);

	auto f = [this](const dlib_vector& v)->double {
		RowVectorXd x(m_cx.size()), y(v.size() - m_cx.size());
		DecomposeOptimizeVector(v, x, y);
		return objective(x, y);
	};

	auto df = [this](const dlib_vector& v)->dlib_vector {
		RowVectorXd x(m_cx.size()), y(v.size() - m_cx.size());
		DecomposeOptimizeVector(v, x, y);
		auto derv = objective_derv(x, y);
		dlib_vector val(derv.size());
		std::copy_n(derv.data(), v.size(), val.begin());
		return val;
	};

	//auto numberic_diff = dlib::derivative(f)(v);
	//auto anaylatic_diff = df(v);

	//std::cout << "numberic derv = " << dlib::trans(numberic_diff) << "anaylatic derv = " << dlib::trans(anaylatic_diff) << std::endl;
	//std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
	//	<< dlib::length(numberic_diff - anaylatic_diff) << std::endl;


	auto result = dlib::find_min //_box_constrained
		(
		dlib::lbfgs_search_strategy(v.size()),
		dlib::objective_delta_stop_strategy(),//.be_verbose(),
		f,
		df,
		//dlib::derivative(f),
		v,
		0//vmin,vmax//,0
		);

	DecomposeOptimizeVector(v, m_cx, m_cy);

	m_currentError = result;
	//cout << result << endl;
	//cout << "joints angles = " << m_cy << endl;
	return m_cy;
}

float StylizedChainIK::Fit(const Eigen::MatrixXf & X, const Eigen::MatrixXf & Y)
{
	return m_gplvm.fit_model(X, Y);
}

float StylizedChainIK::Predict(const Eigen::RowVectorXf & X, Eigen::RowVectorXf & Y)
{
	Y = Apply(X).cast<float>();
	return m_gplvm.get_likelihood_xy(m_cx,m_cy);
}


float Causality::PcaCcaIK::Fit(const Eigen::MatrixXf & X, const Eigen::MatrixXf & Y)
{
	return m_cca.CreateFrom(X, Y, m_xCut, m_yCut);
}

float Causality::PcaCcaIK::Predict(const Eigen::RowVectorXf & X, Eigen::RowVectorXf & Y)
{
	m_cca.Apply(X, Y);
	return 1.0f;
}


float Causality::GaussianProcessIK::Fit(const Eigen::MatrixXf & X, const Eigen::MatrixXf & Y)
{
	return m_gpr.fit_model(X,Y);
}

float Causality::GaussianProcessIK::Predict(const Eigen::RowVectorXf & X, Eigen::RowVectorXf & Y)
{
	RowVectorXd x = X.cast<double>(), y = Y.cast<double>();
	auto lh = m_gpr.get_expectation_from_observation(x, m_obsrCov,&y);
	Y = y.cast<float>();
	return lh;
}

Causality::RelativeLnQuaternionDecoder::~RelativeLnQuaternionDecoder()
{
}

inline void Causality::RelativeLnQuaternionDecoder::Decode(DirectX::Quaternion * rots, const Eigen::RowVectorXd & y)
{
	int n = y.size() / 3;
	Eigen::Vector4f qs;
	XMVECTOR q, qb;
	qs.setZero();
	for (int i = 0; i < n; i++)
	{
		qs.segment<3>(0) = y.segment<3>(i * 3).cast<float>();
		q = XMLoadFloat4A(qs.data());
		q = XMQuaternionExp(q); // revert the log map
		qb = XMLoadA(bases[i]);
		q = XMQuaternionMultiply(qb, q);
		XMStoreA(rots[i], q);
	}
}

inline void Causality::RelativeLnQuaternionDecoder::operator()(DirectX::Quaternion * rots, const Eigen::RowVectorXd & y)
{
	Decode(rots, y);
}

Causality::RelativeLnQuaternionPcaDecoder::~RelativeLnQuaternionPcaDecoder()
{

}

inline void Causality::RelativeLnQuaternionPcaDecoder::DecodePcad(DirectX::Quaternion * rots, const Eigen::RowVectorXd & y)
{
	Eigen::RowVectorXd dy = y * invPcaY + meanY;
	Decode(rots, dy);
}

void Causality::RelativeLnQuaternionPcaDecoder::EncodeJacobi(Eigen::MatrixXd & jacb)
{
	jacb *= pcaY;
}

inline void Causality::RelativeLnQuaternionPcaDecoder::operator()(DirectX::Quaternion * rots, const Eigen::RowVectorXd & y)
{
	DecodePcad(rots, y);
}

Causality::IFeatureDecoder::~IFeatureDecoder()
{
}

void Causality::IFeatureDecoder::EncodeJacobi(Eigen::MatrixXd & jacb)
{
}

Causality::AbsoluteLnQuaternionDecoder::~AbsoluteLnQuaternionDecoder() {}

inline void Causality::AbsoluteLnQuaternionDecoder::operator()(DirectX::Quaternion * rots, const Eigen::RowVectorXd & y)
{
	AbsoluteLnQuaternionDecode(rots, y);
}
