#include "pch.h"
#include "StylizedIK.h"
#pragma warning (push)
#pragma warning (disable : 4297 )
#include <dlib\optimization\optimization.h>
#pragma warning (pop)
#include "Causality\Settings.h"

using namespace Causality;
using namespace std;
using namespace Eigen;
using namespace DirectX;

static const int g_defaultMaxIter = 20;
static const double g_MaxRotationStep = 0.1;

typedef dlib::matrix<double, 0, 1> dlib_vector;
using row_vector_t = StylizedChainIK::row_vector_t;

namespace detail
{
	template <class DerivedX, class DerivedY>
	static dlib_vector ComposeOptimizeVector(const DenseBase<DerivedX> &x, const DenseBase<DerivedY> &y);

	static void DecomposeOptimizeVector(const dlib_vector& v, RowVectorXd &x, RowVectorXd &y);

	template<class DerivedX, class DerivedY>
	inline dlib_vector ComposeOptimizeVector(const DenseBase<DerivedX>& x, const DenseBase<DerivedY>& y)
	{
		dlib_vector v(x.size() + y.size());
		RowVectorXd::Map(v.begin(), x.size()) = x;
		RowVectorXd::Map(v.begin() + x.size(), y.size()) = y;
		return v;
	}

	inline void DecomposeOptimizeVector(const dlib_vector & v, RowVectorXd & x, RowVectorXd & y)
	{
		x = RowVectorXd::Map(v.begin(), x.size());
		y = RowVectorXd::Map(v.begin() + x.size(), y.size());
	}
}

using namespace ::detail;

StylizedChainIK::StylizedChainIK()
{
	reset();
}

StylizedChainIK::StylizedChainIK(size_t n)
	: StylizedChainIK()
{
}

StylizedChainIK::StylizedChainIK(const std::vector<const Joint*>& joints, ArmatureFrameConstView defaultframe)
	: StylizedChainIK()
{
	setChain(joints, defaultframe);
}

StylizedChainIK::~StylizedChainIK()
{
}

void StylizedChainIK::setDecoder(std::unique_ptr<StylizedChainIK::IFeatureDecoder>&& decoder)
{
	m_fpDecoder = move(decoder);
}

void StylizedChainIK::setChain(const std::vector<const Joint*>& joints, ArmatureFrameConstView defaultframe)
{
	resize(joints.size());

	m_chainLength = 0;
	std::vector<DirectX::Quaternion, DirectX::XMAllocator>	rots;
	for (int i = 0; i < joints.size(); i++)
	{
		auto& bone = defaultframe[joints[i]->ID];
		//auto Q = XMLoadA(bone.LclRotation);
		auto V = XMLoadA(bone.LclTranslation);
		//V = XMVector3InverseRotate(V, Q);
		m_bones[i] = V;

		m_chainLength += bone.LclTranslation.Length();
	}

	m_iy = m_gpr.uY;

	m_iy = m_gplvm.uY;
	m_ix.setZero(m_gplvm.latent_dimension());

	m_minY = m_gplvm.Y.colwise().minCoeff() * g_IKLimitMultiplier + m_gplvm.uY;
	m_maxY = m_gplvm.Y.colwise().maxCoeff() * g_IKLimitMultiplier + m_gplvm.uY;

	if (!m_cValiad)
		m_cy = m_iy;
}

void StylizedChainIK::setIKWeight(double weight){m_ikWeight = weight; }
void StylizedChainIK::setMarkovWeight(double weight){ m_markovWeight = weight;}

void StylizedChainIK::setHint(const Eigen::RowVectorXd & y)
{
	m_cy = y;
	m_iy = y;
	m_cValiad = true;
}

double StylizedChainIK::ikDistance(const row_vector_t & y) const
{
	const auto n = m_bones.size();

	rotation_collection_t rotations(n);
	decode(rotations, y);

	XMVECTOR ep = endPosition(rotations);
	Vector3f epf; XMStoreFloat3(epf.data(), ep);

	XMStoreFloat3(epf.data(), ep);

	double ikdis = (epf.cast<double>() - m_goal).squaredNorm() * m_ikWeight / m_chainLength;

	return ikdis;
}

StylizedChainIK::row_vector_t StylizedChainIK::ikDistanceDerivative(const row_vector_t & y) const
{
	const auto n = m_bones.size();

	rotation_collection_t rotations(n);
	jaccobi_collection_t  jaccobis(3 * n);

	decode(rotations, y);

	XMVECTOR ep = endPosition(rotations);
	Vector3f epf; XMStoreFloat3(epf.data(), ep);

	endPositionJaccobiRespectLnQuaternion(rotations, jaccobis);
	MatrixXd jacb(3, 3 * n);
	jacb.leftCols(3 * (n - 1)) = Eigen::Matrix3Xf::Map(&jaccobis[3].x, 3, 3 * (n - 1)).cast<double>();
	jacb.rightCols(3).setZero();

	m_fpDecoder->EncodeJacobi(rotations, jacb); // this should be empty function

	// IK term derv
	RowVectorXd ikderv = (2.0 * m_ikWeight / m_chainLength * (epf.cast<double>() - m_goal)).transpose() * jacb;

	return ikderv;
}

double StylizedChainIK::limitDistance(const row_vector_t & y) const
{
	return  m_ikLimitWeight * ((m_minY - y).cwiseMax(y - m_maxY)).cwiseMax(0).cwiseAbs2().sum();
}

row_vector_t StylizedChainIK::limitDistanceDerivative(const row_vector_t & y) const
{
	return 2.0 * m_ikLimitWeight * ((y - m_minY).cwiseMin(0) + (y - m_maxY).cwiseMax(0));
}

double StylizedChainIK::objective(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y)
{
	double ikdis = ikDistance(y);

	double iklimdis = limitDistance(y);

	double markovdis = ((y - m_cy).cwiseAbs2().array() / m_cyNorm.array()).sum() / (double)m_cy.size() * m_markovWeight;

	double fitlikelihood = ((y.array() /** m_wy.array()*/ - m_ey.array()).cwiseAbs2() / m_eyNorm.array()).sum() * (0.5 * m_segmaX);
	//double fitlikelihood = m_gpr.get_likelihood_xy(x, y) * g_StyleLikelihoodTermWeight;

	return ikdis + fitlikelihood + markovdis;//+iklimdis;
}

RowVectorXd StylizedChainIK::objective_derv(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y)
{
	RowVectorXd derv(x.size() + y.size());

	// IK term derv
	RowVectorXd ikderv = ikDistanceDerivative(y);

	RowVectorXd iklimderv = limitDistanceDerivative(y);

	// Markov progation derv
	RowVectorXd markovderv = 2.0 * m_markovWeight * ((y - m_cy).array() / m_cyNorm.array()) / (double)m_cy.size();
	//markovderv.setZero();

	RowVectorXd animLkderv = (y.array()/* * m_wy.array()*/ - m_ey.array()) / m_eyNorm.array() * (m_segmaX); //m_gpr.get_likelihood_xy_derivative(x, y) * g_StyleLikelihoodTermWeight;


	derv.segment(x.size(), y.size()) = ikderv + markovderv;// + gplvm.likelihood_xy_derv;
	derv.segment(0, x.size()).setZero();
	derv.segment(x.size(), y.size()) += animLkderv;
	//derv.segment(x.size(), y.size()) += iklimderv;

	return derv;//derv
}


// Filter alike interface
// reset history data

void StylizedChainIK::reset()
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
	m_maxIter = g_defaultMaxIter;
}

// return the joints rotation vector
RowVectorXd StylizedChainIK::apply(const Vector3d & goal, const DirectX::Quaternion & baseRotation)
{
	setGoal(goal);
	m_baseRot = baseRotation;

	if (!m_cValiad)
	{
		m_cx = m_goal;
		m_cy = m_iy;
		m_cValiad = true;
	}

	RowVectorXd hint_y;
	double lk = m_gpr.get_ey_on_x(m_goal, &hint_y);
	lk = exp(-lk);

	rotation_collection_t rotations(m_bones.size());
	decode(rotations, m_cy);
	Vector3 cyep = endPosition(rotations);
	Vector3d cyepmap = Vector3f::Map(&cyep.x).cast<double>();
	Vector3d cydiff = m_goal - cyepmap;


	decode(rotations, hint_y);
	Vector3 ep = endPosition(rotations);
	Vector3d epmap = Vector3f::Map(&ep.x).cast<double>();
	Vector3d diff = m_goal - epmap;

	// too many relative error
	double eydiffn = diff.stableNorm() /*/ m_goal.stableNorm()*/;
	double cydiffn = cydiff.stableNorm() /*/ m_goal.stableNorm()*/;

	//if (eydiffn > 0.05 && cydiffn > 0.05)
	//{
	//	hint_y = m_iy;
	//} else 
	if (cydiffn < eydiffn || (m_cy - hint_y).norm() > g_MaxRotationStep * m_ey.size())
	{
		//? This condition are NEVER triggered!!! HOW?????????????
		hint_y = m_cy;
		if (g_EnableDebugLogging >= 2)
			std::cout << "Use cy instead of ey" << endl;
	}

	//hint_y = m_cy;
	hint_y = hint_y.cwiseMin(m_maxY).cwiseMax(m_minY);
	//if ((hint_y.array() > m_maxY.array()).any() || (hint_y.array() < m_minY.array()).any())
	//{
	//	hint_y = m_iy;
	//	if (g_EnableDebugLogging >= 2)
	//		std::cout << "Use iy because out of bound" << endl;
	//}



	//m_meanLk = (m_meanLk * m_counter + lk) / (m_counter + 1);
	//++m_counter;
	//if (lk < 0.001 * m_meanLk) // low possibility prediction, use previous frame's data instead
	//	hint_y = m_cy;
	//hint_y = m_cy;

	m_cx = m_goal;
	return apply(m_goal, hint_y.transpose().eval());

}

void StylizedChainIK::setGoal(const Eigen::Vector3d & goal)
{
	// make the goal achiveable 
	if (goal.norm() <= m_chainLength)
		m_goal = goal;
	else
		m_goal = goal * (m_chainLength / goal.norm());

}

Eigen::RowVectorXd StylizedChainIK::apply(const Eigen::Vector3d & goal, const Eigen::Vector3d & goal_velocity, const DirectX::Quaternion & baseRotation)
{
	setGoal(goal);
	m_baseRot = baseRotation;

	RowVectorXd x(1, 6);
	x.segment<3>(0) = m_goal;
	x.segment<3>(3) = goal_velocity;

	RowVectorXd hint_y;
	double lk = m_gpr.get_ey_on_x(x, &hint_y);
	//hint_y.array() /= m_wy.array();
	lk = exp(-lk);

	if (!m_cValiad)
	{
		m_cx = m_goal;
		m_cy = hint_y;
		m_cValiad = true;
	}

	m_meanLk = (m_meanLk * m_counter + lk) / (m_counter + 1);
	++m_counter;

	//if (lk < 0.001 * m_meanLk) // low possibility prediction, use previous frame's data instead
	hint_y = m_cy;

	m_cx = x;

	return apply(m_goal, hint_y.transpose().eval());//
}

Eigen::RowVectorXd Causality::StylizedChainIK::apply(const Eigen::Vector3d & goal, const Eigen::VectorXd & hint_y)
{

	auto v = ComposeOptimizeVector(m_cx, hint_y);
	auto backupV = v;
	auto xmin = m_cx; auto xmax = m_cx;
	xmin.setConstant(-10000);
	xmax.setConstant(-10000);

	auto vmin = ComposeOptimizeVector(m_cx, m_minY);
	auto vmax = ComposeOptimizeVector(m_cx, m_maxY);

	double d2lnvarx = m_gpr.get_ey_on_x(m_cx, &m_ey);
	double varX = exp(d2lnvarx / (double)m_ey.cols() * 2.0);
	m_segmaX = m_styleWeight / varX;

	m_cyNorm = .01 + m_cy.array().cwiseAbs2();
	//m_eyNorm = .1 + m_ey.array().cwiseAbs2(); //
	m_eyNorm.setOnes(m_ey.size());//

	bool verbose = false;

	auto f = [this,&verbose](const dlib_vector& v)->double {
		RowVectorXd x(m_cx.size()), y(v.size() - m_cx.size());
		DecomposeOptimizeVector(v, x, y);
		double value = objective(x, y);
		if (verbose)
			cout << "x = " << dlib::trans(v) << " value == " << value << endl;
		return value;
	};

	auto df = [this,&verbose](const dlib_vector& v)->dlib_vector {
		RowVectorXd x(m_cx.size()), y(v.size() - m_cx.size());
		DecomposeOptimizeVector(v, x, y);
		auto derv = objective_derv(x, y);
		dlib_vector val(derv.size());
		std::copy_n(derv.data(), v.size(), val.begin());
		if (verbose)
			cout << "x = " << dlib::trans(v) << " gradiatn == " << derv << endl;
		return val;
	};
	auto search_strategy = dlib::bfgs_search_strategy();
	
	//auto numberic_diff = dlib::derivative(f)(v);
	//auto anaylatic_diff = df(v);

	//std::cout << "numberic derv = " << dlib::trans(numberic_diff) << "anaylatic derv = " << dlib::trans(anaylatic_diff) << std::endl;
	//std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
	//	<< dlib::length(numberic_diff - anaylatic_diff) << std::endl;

	double result;

	try
	{
		result = dlib::find_min_box_constrained
			(
				//dlib::lbfgs_search_strategy(v.size()),
				search_strategy,
				dlib::gradient_norm_stop_strategy(1e-3, m_maxIter),//.be_verbose(),
				f,
				df,
				//dlib::derivative(f),
				v,
				vmin,vmax
				);

		DecomposeOptimizeVector(v, m_cx, m_cy);
	}
	catch (const std::exception&)
	{
		// skip this frame if failed to optimze
		result = 1.0;
	}

	//if (result / m_goal.norm() > 0.05)
	//{
	//	v = ComposeOptimizeVector(m_cx, m_iy);
	//}



	//m_cy = m_ey;
	m_currentError = result;

	rotation_collection_t rotations(m_bones.size());
	decode(rotations, m_cy);
	Vector3 ep = ep = endPosition(rotations);

	if (result > 0.01)
	{
		if (g_EnableDebugLogging >= 2)
		{
			cout << "IK failed : initial x == " << dlib::trans(backupV) << endl;
			if (g_EnableDebugLogging >= 3)
			{
				v = backupV;
				verbose = true;
				result = dlib::find_min //_box_constrained
					(
						search_strategy,
						dlib::gradient_norm_stop_strategy(1e-3, m_maxIter).be_verbose(),
						f, df, v, 0//vmin,vmax//,0
						);
			}
		}
	}
	if (g_EnableDebugLogging >= 2)
	{
		cout << "pred = " << m_segmaX << " error = " << result << " | goal = (" << goal.transpose() << ") achived = (" << ep << ')' << endl;
		cout << "joints angles = " << m_cy << endl;
	}

	return m_cy;
}

double StylizedChainIK::objective_xy(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y)
{
	double ikdis = 0;
	ikdis = ikDistance(y);
	double limitdis = 0;
	limitdis = limitDistance(y);

	double stylik = 0;
	stylik = m_styleWeight / (double)y.cols() * m_gplvm.get_likelihood_xy(x, y);

	return ikdis + stylik;
}

RowVectorXd StylizedChainIK::objective_xy_derv(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y)
{
	RowVectorXd derv(x.size() + y.size());
	derv.setZero();

	// stylik gradiant
	derv = m_styleWeight / (double)y.cols() * m_gplvm.get_likelihood_xy_derivative(x, y);
	derv.segment(x.size(), y.size()) += ikDistanceDerivative(y);
	derv.segment(x.size(), y.size()) += limitDistanceDerivative(y);

	return derv;
}

void StylizedChainIK::decode(array_view<DirectX::Quaternion> rots, const row_vector_t & y) const
{
	m_fpDecoder->Decode(rots, y.cast<float>());
	for (int i = m_bones.size() - 1; i > 0; i--)
		rots[i] = rots[i - 1];
	rots[0] = m_baseRot;
}

void StylizedChainIK::encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ row_vector_t& y) const
{
	std::vector<DirectX::Quaternion, DirectX::XMAllocator> tempRots;
	for (int i = 0; i < m_bones.size() - 1; i++)
		tempRots[i] = rots[i + 1];
	tempRots[m_bones.size() - 1] = Quaternion::Identity;
	Eigen::RowVectorXf fy(y.size());
	m_fpDecoder->Encode(rots, fy);
	y = fy.cast<double>();
}


row_vector_t StylizedChainIK::solve(const vector3_t & goal, const vector3_t & goal_vel, const DirectX::Quaternion & baseRotation)
{
	dlib_vector v;
	if (m_cValiad)
		v = ComposeOptimizeVector(m_cx, m_cy);
	else
	{
		v = ComposeOptimizeVector(m_ix, m_iy);
		m_cx = m_ix; m_cy = m_iy;
	}

	m_baseRot = baseRotation;
	setGoal(goal);

	auto f = [this](const dlib_vector& v)->double {
		RowVectorXd x(m_cx.size()), y(v.size() - m_cx.size());
		DecomposeOptimizeVector(v, x, y);
		return objective_xy(x, y);
	};

	auto df = [this](const dlib_vector& v)->dlib_vector {
		RowVectorXd x(m_cx.size()), y(v.size() - m_cx.size());
		DecomposeOptimizeVector(v, x, y);
		auto derv = objective_xy_derv(x, y);
		dlib_vector val(derv.size());
		std::copy_n(derv.data(), v.size(), val.begin());
		return val;
	};

	//auto numberic_diff = dlib::derivative(f)(v);
	//auto anaylatic_diff = df(v);

	//std::cout << "numberic derv = " << dlib::trans(numberic_diff) << "anaylatic derv = " << dlib::trans(anaylatic_diff) << std::endl;
	//std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
	//	<< dlib::length(numberic_diff - anaylatic_diff) << std::endl;

	double result;

	try
	{
		result = dlib::find_min //_box_constrained
			(
				dlib::cg_search_strategy(),
				dlib::objective_delta_stop_strategy(1e-2, m_maxIter),//.be_verbose(),
				f, df,
				v,
				-1e5
				);

		DecomposeOptimizeVector(v, m_cx, m_cy);
		m_cValiad = true;

	}
	catch (const std::exception& exc)
	{
		// skip this frame if failed to optimze
		result = 1.0;
		m_cValiad = false;
		cout << "[Error] S-IK failed : " << exc.what() << std::endl;
	}

	rotation_collection_t rotations(m_bones.size());
	decode(rotations, m_cy);
	Vector3 ep = endPosition(rotations);

	if (g_EnableDebugLogging >= 2)
	{
		cout << "pred = " << m_segmaX << " error = " << result << " | goal = (" << goal.transpose() << ") achived = (" << ep << ')' << endl;
		cout << "joints angles = " << m_cy << endl;
	}

	return m_cy;
}
