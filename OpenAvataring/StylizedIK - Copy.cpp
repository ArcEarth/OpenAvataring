#include "pch.h"
#include "StylizedIK.h"
#include <unsupported\Eigen\NonLinearOptimization>
#include "Causality\Settings.h"

using namespace Causality;
using namespace std;
using namespace Causality::Math;
using namespace Eigen;

namespace Internal
{
	// Generic functor
	template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
	struct Functor
	{
		typedef _Scalar Scalar;
		enum {
			InputsAtCompileTime = NX,
			ValuesAtCompileTime = NY
		};
		typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
		typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
		typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

		int m_inputs, m_values;

		Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
		Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

		int inputs() const { return m_inputs; }
		int values() const { return m_values; }

	};
}


//typedef dlib::matrix<double, 0, 1> dlib_vector;

//typedef VectorXd InputParamVector;
//template <class DerivedX, class DerivedY>
//InputParamVector ComposeOptimizeVector(const DenseBase<DerivedX> &x, const DenseBase<DerivedY> &y);
//
//void DecomposeOptimizeVector(const InputParamVector& v, RowVectorXd &x, RowVectorXd &y);
//
//template<class DerivedX, class DerivedY>
//inline InputParamVector ComposeOptimizeVector(const DenseBase<DerivedX>& x, const DenseBase<DerivedY>& y)
//{
//	InputParamVector v(x.size() + y.size());
//	RowVectorXd::Map(v.data(), x.size()) = x;
//	RowVectorXd::Map(v.data() + x.size(), y.size()) = y;
//	return v;
//}
//
//void DecomposeOptimizeVector(const InputParamVector & v, RowVectorXd & x, RowVectorXd & y)
//{
//	x = RowVectorXd::Map(v.data(), x.size());
//	y = RowVectorXd::Map(v.data() + x.size(), y.size());
//}

struct StylizedChainIK::OptimizeFunctor : public ::Internal::Functor<float>
{
	const StylizedChainIK&			sik;
	size_t							n;
	IFeatureDecoder&												m_decoder;
	Vector3						                                    m_goal;
	float						                                    m_limitPanalty;
	Eigen::Map<const Eigen::VectorXf>                               m_min;
	Eigen::Map<const Eigen::VectorXf>                               m_max;

	Eigen::VectorXf					                                m_ref;
	float							                                m_refWeights;
	bool							                                m_useRef;

	mutable std::vector<DirectX::Quaternion, DirectX::XMAllocator>	m_rots;
	mutable std::vector<DirectX::Vector3, DirectX::XMAllocator>		m_jacvectos;
	mutable OptimzeJacobiType										m_jacb;

	OptimizeFunctor(const StylizedChainIK& _sik)
		: sik(_sik), n(_sik.size()),
		::Internal::Functor<float>(3 * _sik.size(), 3 + 3 * _sik.size()),
		m_rots(_sik.size()),
		m_jacb(3,3*_sik.size()),
		m_ref(_sik.size() * 3),
		m_min(&_sik.m_boneMinLimits[0].x, 3 * _sik.size()),//? WRONG!!!
		m_max(&_sik.m_boneMaxLimits[0].x, 3 * _sik.size()),//? WRONG!!!
		m_limitPanalty(1000.0f),
		m_refWeights(.0f),
		m_useRef(false),
		m_decoder(*_sik.m_fpDecoder)
	{
	}

	void setGoal(const Vector3 & goal)
	{
		m_goal = goal;
	}

	template <class Derived>
	void setReference(const Eigen::DenseBase<Derived>& refernece, float referenceWeight)
	{
		m_ref = refernece;
		m_refWeights = referenceWeight;
		m_useRef = true;
	}

	void disableRef()
	{
		m_useRef = false;
		m_refWeights = .0f;
	}

	int operator()(const InputType &x, ValueType& fvec) const {
		m_decoder.Decode(m_rots, x);

		Vector3 v = sik.endPosition(m_rots);
		v -= m_goal;
		fvec.setZero();
		fvec.head<3>() = Eigen::Vector3f::Map(&v.x);

		// limit-exceed panelaty
		auto limpanl = fvec.tail(x.size());
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] < m_min[i])
				limpanl[i] = m_limitPanalty*(x[i] - m_min[i])*(x[i] - m_min[i]);
			else if (x[i] > m_max[i])
				limpanl[i] = m_limitPanalty*(x[i] - m_max[i])*(x[i] - m_max[i]);
		}

		if (m_useRef)
		{
			limpanl += m_refWeights *(x - m_ref);
		}

		return 0;
	}

	int df(const InputType &x, JacobianType& fjac) {
		m_decoder.Decode(m_rots, x);

		fjac.setZero();

		m_jacb.resize(3, 3 * n);
		sik.endPositionJaccobiRespectEuler(m_rots, 
			array_view<Vector3>(reinterpret_cast<Vector3*>(m_jacb.data()),3*n));

		m_decoder.EncodeJacobi(m_rots, m_jacb);

		fjac.topRows<3>() = m_jacb;//Eigen::Matrix3Xf::Map(&m_jac[0].x, 3, 3 * n);

		// limit-exceed panelaty
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] < m_min[i])
				fjac(3 + i, i) = m_limitPanalty * (x[i] - m_min[i]);
			else if (x[i] > m_max[i])
				fjac(3 + i, i) = m_limitPanalty * (x[i] - m_max[i]);

			if (m_useRef)
			{
				fjac(3 + i, i) += m_refWeights;
			}
		}

		return 0;
	}
};


StylizedChainIK::StylizedChainIK()
{
	reset();
}

StylizedChainIK::StylizedChainIK(size_t n)
	: ChainInverseKinematics(n)
{
	reset();
}

StylizedChainIK::StylizedChainIK(const std::vector<const Joint*>& joints, ArmatureFrameConstView defaultframe)
	: StylizedChainIK()
{
	setChain(joints, defaultframe);
}

StylizedChainIK::~StylizedChainIK()
{

}

void StylizedChainIK::setDecoder(std::unique_ptr<IFeatureDecoder>&& decoder)
{
	m_fpDecoder = move(decoder);
}

void StylizedChainIK::setChain(const std::vector<const Joint*>& joints, ArmatureFrameConstView defaultframe)
{
	using namespace DirectX;
	using namespace Eigen;

	resize(joints.size());

	m_chainLength = 0;
	std::vector<DirectX::Quaternion, DirectX::XMAllocator>	rots;
	for (int i = 0; i < joints.size(); i++)
	{
		auto& bone = defaultframe[joints[i]->ID];
		auto Q = XMLoadA(bone.LclRotation);
		auto V = XMLoadA(bone.LclTranslation);
		V = XMVector3InverseRotate(V,Q);
		m_bones[i] = V;

		m_chainLength += bone.LclTranslation.Length();
	}

	//m_fpDecoder->Encode(rots, m_iy);

	m_iy = m_gplvm.uY.cast<float>();
	if (m_iy.stableNorm() < 0.01)
	{
		m_iy.setConstant(0.1f);
	}
	m_pFunctor.reset(new OptimizeFunctor(*this));
}

void StylizedChainIK::setIKWeight(double weight)
{
	m_ikWeight = weight;
}

void StylizedChainIK::setMarkovWeight(double weight)
{
	m_markovWeight = weight;
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
	m_fpDecoder.reset(new AbsoluteLnQuaternionDecoder());
}


// return the joints rotation vector

const StylizedChainIK::OptimzeVectorType & StylizedChainIK::apply(const Vector3 & goal, const DirectX::Quaternion & baseRotation)
{
	Vector3d agoal = Vector3f::Map(&goal.x).cast<double>();

	if (!m_cValiad)
	{
		m_cx = agoal;
		m_cy = m_iy;
		m_cValiad = true;
	}

	// calculate achievable goal
	if (agoal.norm() <= m_chainLength)
		agoal = agoal;
	else
		agoal = agoal * (m_chainLength / agoal.norm());

	// caculate style reference
	m_cx = agoal;

	m_segmaX = m_gplvm.get_expectation_and_likelihood(m_cx, &m_ey);
	m_segmaX = exp(2 / m_cy.cols() * m_segmaX);

	//Eigen::NumericalDiff<OptimizeFunctor> functor(*this);
	auto& functor = *m_pFunctor;
	XMVECTOR vg = XMVector3InverseRotate(goal, baseRotation);
	functor.setGoal(vg);
	//functor.setReference(m_ey.cast<float>(), 0);// m_styleWeight * m_segmaX);

	Eigen::LevenbergMarquardt<OptimizeFunctor, float> lm(functor);
	lm.parameters.maxfev = m_maxItrs;
	lm.parameters.xtol = m_tol;
	lm.parameters.ftol = m_tol;
	lm.parameters.gtol = m_tol;

	VectorXf x = m_cy;

	std::vector<DirectX::Quaternion> rots(this->size());
	m_fpDecoder->Decode(rots, x);
	ChainInverseKinematics::solve(vg, rots);
	Vector3 achived = endPosition(rots);
	cout << "goal = " << Vector3(vg) << " achived = " << achived << endl;

	m_fpDecoder->Encode(rots,x);
	//auto code = lm.minimize(x);
	//cout << code << ':' << lm.iter << ':' << lm.fnorm << endl;
	m_currentError = lm.fnorm;
	m_cx = agoal;
	m_cy = x;

	return m_cy;
}

const StylizedChainIK::OptimzeVectorType & StylizedChainIK::apply(const Vector3 & goal, const Vector3 & goal_vel, const DirectX::Quaternion & baseRotation)
{
	return m_cy;
	// TODO: insert return statement here
}


RelativeLnQuaternionDecoder::~RelativeLnQuaternionDecoder()
{}
AbsoluteLnQuaternionDecoder::~AbsoluteLnQuaternionDecoder()
{}
AbsoluteEulerAngleDecoder::~AbsoluteEulerAngleDecoder()
{}
RelativeLnQuaternionPcaDecoder::~RelativeLnQuaternionPcaDecoder()
{}

void AbsoluteLnQuaternionDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
{
	int n = rots.size();

	Eigen::Vector4f qs;
	XMVECTOR q;
	qs.setZero();
	for (int i = 0; i < n; i++)
	{
		qs.segment<3>(0) = x.segment<3>(i * 3).cast<float>();
		q = XMLoadFloat4A(qs.data());
		q = XMQuaternionExp(q); // revert the log map
		XMStoreA(rots[i], q);
	}
}

void AbsoluteLnQuaternionDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
{
}

void AbsoluteLnQuaternionDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
{}


void AbsoluteEulerAngleDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & y)
{
	int n = rots.size();

	Eigen::Vector4f qs;
	XMVECTOR q;
	qs.setZero();
	for (int i = 0; i < n; i++)
	{
		qs.segment<3>(0) = y.segment<3>(i * 3).cast<float>();
		q = XMLoadFloat4A(qs.data());
		q = XMQuaternionRotationRollPitchYawFromVector(q); // revert the log map
		XMStoreA(rots[i], q);
	}
}

void AbsoluteEulerAngleDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
{
	int n = rots.size();

	Eigen::Vector4f qs;
	XMVECTOR q;
	qs.setZero();
	x.resize(n * 3);
	for (int i = 0; i < n; i++)
	{
		q = XMLoad(rots[i]);
		q = XMQuaternionEulerAngleYawPitchRoll(q); // Decompsoe in to euler angle
		XMStoreFloat4(qs.data(), q);
		x.segment<3>(i * 3) = qs.head<3>();
	}
}

void AbsoluteEulerAngleDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
{
}

void RelativeEulerAngleDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
{
	int n = rots.size();

	Eigen::Vector4f qs;
	XMVECTOR q, qb;
	qs.setZero();
	for (int i = 0; i < n; i++)
	{
		qs.segment<3>(0) = x.segment<3>(i * 3).cast<float>();
		q = XMLoadFloat4A(qs.data());
		q = XMQuaternionRotationRollPitchYawFromVector(q); // revert the log map
		qb = XMLoadA(bases[i]);
		q = XMQuaternionMultiply(qb, q);
		XMStoreA(rots[i], q);
	}
}

void RelativeEulerAngleDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
{
	int n = rots.size();
	Eigen::Vector4f qs;
	XMVECTOR q ,qb;
	qs.setZero();
	x.resize(n * 3);
	for (int i = 0; i < n; i++)
	{
		q = XMLoad(rots[i]);
		qb = XMLoadA(bases[i]);
		qb = XMQuaternionInverse(qb);
		q = XMQuaternionMultiply(qb, q);
		q = XMQuaternionEulerAngleYawPitchRoll(q); // Decompsoe in to euler angle
		XMStoreFloat4(qs.data(), q);
		x.segment<3>(i * 3) = qs.head<3>();
	}
}

void RelativeEulerAngleDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
{
}

void RelativeLnQuaternionDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
{
	int n = rots.size();
	Eigen::Vector4f qs;
	XMVECTOR q, qb;
	qs.setZero();
	for (int i = 0; i < n; i++)
	{
		qs.segment<3>(0) = x.segment<3>(i * 3).cast<float>();
		q = XMLoadFloat4A(qs.data());
		q = XMQuaternionExp(q); // revert the log map
		qb = XMLoadA(bases[i]);
		q = XMQuaternionMultiply(qb, q);
		XMStoreA(rots[i], q);
	}
}

void RelativeLnQuaternionDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
{
}

void RelativeLnQuaternionDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
{
	assert(false);
}
void RelativeLnQuaternionPcaDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
{
	VectorType dy = (x.cast<double>() * invPcaY + meanY).cast<float>();
	RelativeLnQuaternionDecoder::Decode(rots, dy);
}

void RelativeLnQuaternionPcaDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
{
	RelativeLnQuaternionDecoder::Encode(rots, x);
	RowVectorXd dx = x.transpose().cast<double>();
	dx -= meanY;
	dx *= pcaY;
	x = dx.transpose().cast<float>();
}

void RelativeLnQuaternionPcaDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
{
	jacb *= pcaY.cast<float>();
}

void RelativeEulerAnglePcaDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
{
	VectorType dy = (x.transpose().cast<double>() * invPcaY + meanY).cast<float>().transpose();
	RelativeEulerAngleDecoder::Decode(rots, dy);
}

RelativeEulerAnglePcaDecoder::~RelativeEulerAnglePcaDecoder()
{
}

void RelativeEulerAnglePcaDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
{
	RelativeEulerAngleDecoder::Encode(rots, x);
	RowVectorXd dx = x.transpose().cast<double>();
	dx -= meanY;
	dx *= pcaY;
	x = dx.transpose().cast<float>();
}

void RelativeEulerAnglePcaDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
{
	jacb *= pcaY.cast<float>();
}

StylizedChainIK::IFeatureDecoder::~IFeatureDecoder()
{
}

Causality::RelativeEulerAngleDecoder::~RelativeEulerAngleDecoder()
{
}