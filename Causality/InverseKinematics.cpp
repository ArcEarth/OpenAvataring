#include "pch_bcl.h"
#include "InverseKinematics.h"
#include <Eigen\Core>
#include <unsupported\Eigen\NonLinearOptimization>

using namespace Causality;

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
namespace Causality
{
	void DecodeRotations(Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> &rotations)
	{
		int n = rotations.size();
		for (int i = 0; i < n; i++)
		{
			auto euler = x.segment<3>(i * 3);
			rotations[i] = DirectX::XMQuaternionRotationRollPitchYaw(euler[0], euler[1], euler[2]);
		}
	}

	void EncodeRotations(Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> &rotations)
	{
		int n = rotations.size();
		for (int i = 0; i < n; i++)
		{
			auto euler = x.segment<3>(i * 3);
			auto dxeuler = DirectX::XMQuaternionEulerAngleYawPitchRoll(rotations[i]);
			DirectX::XMStoreFloat3(euler.data(), dxeuler);
		}
	}
}

struct ChainInverseKinematics::OptimizeFunctor : public Internal::Functor<float>
{
	const ChainInverseKinematics&	ik;
	size_t							n;
	Vector3						                                    m_goal;
	float						                                    m_limitPanalty;
	Eigen::Map<const Eigen::VectorXf>                               m_min;
	Eigen::Map<const Eigen::VectorXf>                               m_max;

	Eigen::VectorXf					                                m_ref;
	float							                                m_refWeights;
	bool							                                m_useRef;

	mutable std::vector<DirectX::Quaternion, DirectX::XMAllocator>	m_rots;
	mutable std::vector<DirectX::Vector3,	DirectX::XMAllocator>	m_jac;

	void fillRotations(const InputType &x) const
	{
		assert(x.size() == 3 * n);
		for (int i = 0; i < n; i++)
		{
			auto euler = x.segment<3>(i * 3);
			m_rots[i] = DirectX::XMQuaternionRotationRollPitchYaw(euler[0], euler[1], euler[2]);
		}
	}

	OptimizeFunctor(const ChainInverseKinematics& _ik, const Vector3 & goal)
		: ik(_ik), n(_ik.size()),
		Internal::Functor<float>(3 * _ik.size(), 3 + 3 * _ik.size()),
		m_rots(_ik.size()),
		m_jac(_ik.size() * 3),
		m_goal(goal),
		m_ref(_ik.size() * 3),
		m_min(&_ik.m_boneMinLimits[0].x, 3 * _ik.size()),
		m_max(&_ik.m_boneMaxLimits[0].x, 3 * _ik.size()),
		m_limitPanalty(1000.0f),
		m_refWeights(.0f),
		m_useRef(false)
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
		fillRotations(x);
		Vector3 v = ik.endPosition(m_rots);
		v -= m_goal;
		fvec.setZero();
		fvec.head<3>() = Eigen::Vector3f::Map(&v.x);

		// limit-exceed panelaty
		auto limpanl = fvec.tail(x.size());
		for (int i = 0; i < 3 * n; i++)
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
		fillRotations(x);
		fjac.setZero();

		ik.endPositionJaccobiRespectEuler(m_rots, m_jac);

		auto jacb = fjac.topRows<3>();
		jacb = Eigen::Matrix3Xf::Map(&m_jac[0].x, 3, 3 * n);

		// limit-exceed panelaty
		for (int i = 0; i < 3 * n; i++)
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

		//fjac.topRows<3>() = m_jac;
		return 0;
	}
};

ChainInverseKinematics::ChainInverseKinematics(size_t n)
	: ChainInverseKinematics()
{
	resize(n);
}

ChainInverseKinematics::ChainInverseKinematics()
{
	m_tol = 5e-4;
	m_maxItrs = 200;
}

void ChainInverseKinematics::resize(size_t n)
{
	m_bones.resize(n);
	m_boneMinLimits.resize(n);
	m_boneMaxLimits.resize(n);
	m_jointWeights.resize(n);

	using namespace DirectX;
	Vector3 dlim(XM_PIDIV2, XM_PI, XM_PI);
	for (int i = 0; i < n; i++)
	{
		m_boneMinLimits[i] = -dlim;
		m_boneMaxLimits[i] = dlim;
		m_jointWeights[i] = Vector3(1.0f);
	}
}

void ChainInverseKinematics::computeJointWeights()
{
	using namespace Math;
	XMVECTOR V;
	XMVECTOR LV = XMVectorZero();
	for (int i = m_bones.size()-1; i >= 0; --i)
	{
		V = XMLoadA(m_bones[i]);
		V = XMVector3Length(V);
		LV += V;
		m_jointWeights[i] = LV;
	}
}

// Jaccobbi from a rotation radius vector (r) respect to a small rotation dr = (drx,dry,drz) in global reference frame
void ChainInverseKinematics::jacobbiRespectAxisAngle(Matrix4x4 & j, const float * r)
{
	j._11 = 0, j._12 = r[2], j._13 = -r[1];
	j._21 = -r[2], j._22 = 0, j._23 = r[0];
	j._31 = r[1], j._32 = -r[0], j._33 = 0;
}

XMMATRIX XM_CALLCONV ChainInverseKinematics::jacobbiTransposeRespectEuler(const Vector3 & rv, const Vector3& euler, FXMVECTOR globalRot)
{
	using namespace DirectX;

	// Jaccobi to Euler 
	// J = Ry*Jy + Ry*Rx*Jx + Ry*Rx*Rz*Jz

	XMVECTOR V;
	XMMATRIX MJ;
	XM_ALIGNATTR Vector4 v, r = rv;

	XMVECTOR Q = globalRot;
	XMVECTOR lQ;

	lQ = XMQuaternionRotationRoll(euler.z);
	Q = XMQuaternionMultiply(lQ, Q);
	//Q = XMQuaternionRotationRollPitchYaw(euler.x, euler.y, euler.z);
	//Q = XMQuaternionMultiply(globalRot, Q);
	v.x = -r.y;
	v.y = r.x;
	v.z = 0;
	V = XMLoadA(v);
	V = XMVector3Rotate(V, Q);
	MJ.r[2] = V;
	r = XMVector3Rotate(r, lQ);

	lQ = XMQuaternionRotationPatch(euler.x);
	Q = XMQuaternionMultiply(lQ,Q);
	//Q = XMQuaternionRotationRollPitchYaw(euler.x, euler.y, 0);
	//Q = XMQuaternionMultiply(globalRot, Q);
	v.x = 0;
	v.y = -r.z;
	v.z = r.y;
	V = XMLoadA(v);
	V = XMVector3Rotate(V, Q);
	MJ.r[0] = V;
	r = XMVector3Rotate(r, lQ);

	lQ = XMQuaternionRotationYaw(euler.y);
	Q = XMQuaternionMultiply(lQ, Q);
	v.x = r.z;
	v.y = 0;
	v.z = -r.x;
	V = XMLoadA(v);
	V = XMVector3Rotate(V, Q);
	MJ.r[1] = V;

	//MJ = XMMatrixTranspose(MJ);
	return MJ;
}

XMVECTOR Causality::ChainInverseKinematics::endPosition(array_view<const Quaternion> rotations) const
{
	using namespace DirectX;

	const auto n = m_bones.size();

	XMVECTOR q, t, gt, gq;
	Eigen::Vector4f qs;
	qs.setZero();

	gt = XMVectorZero();

	for (int i = n - 1; i >= 0; i--)
	{
		q = XMLoadA(rotations[i]);
		t = XMLoadA(m_bones[i]);
		gt += t;
		gt = XMVector3Rotate(gt, q);
	}

	return gt;
}

// rotations must be aligned

void ChainInverseKinematics::endPositionJaccobiRespectEuler(array_view<const Quaternion> rotations, array_view<Vector3> jacb) const
{
	using namespace Eigen;
	using namespace DirectX;
	const auto n = m_bones.size();
	//jacb.resize(3, 3 * n);

	// Chain Position Vectors
	std::vector<DirectX::Vector4, DirectX::XMAllocator>	rad(n);

	XMVECTOR q, t, gt, gq;

	gt = XMVectorZero();
	t = XMVectorZero();

	for (int i = n - 1; i >= 0; i--)
	{
		q = XMLoadA(rotations[i]);
		t = XMLoadA(m_bones[i]);
		gt += t;
		rad[i] = gt;
		gt = XMVector3Rotate(gt, q);
	}

	//XMMATRIX rot;
	XM_ALIGNATTR Matrix4x4 jac;
	jac._11 = jac._22 = jac._33 = 0;
	jac._41 = jac._42 = jac._43 = jac._44 = jac._14 = jac._24 = jac._34 = 0;

	gq = XMQuaternionIdentity();
	for (int i = 0; i < n; i++)
	{
		q = XMLoadA(rotations[i]);
		Vector3 eular = XMQuaternionEulerAngleYawPitchRoll(q);

		auto& r = reinterpret_cast<Vector3&>(rad[i]);
		jac = jacobbiTransposeRespectEuler(r, eular, gq);

		// jacb.block<3,3>(0,3*i) = (float3x3)rot.transpose();
		for (int j = 0; j < 3; j++)
			jacb[i * 3 + j] = Vector3(jac.m[j]);
		//for (int j = 0; j < 3; j++)
		//{
		//	XMStoreFloat3(jacb.col(i * 3 + j).data(), rot.r[j]);
		//}

		gq = XMQuaternionMultiply(q,gq);
	}

	//return jacb;
}

bool XM_CALLCONV ChainInverseKinematics::solve(FXMVECTOR goal, array_view<Quaternion> rotations) const
{
	auto n = m_bones.size();
	Eigen::VectorXf x(n * 3);
	EncodeRotations(x, rotations);
	//m_boneMinLimits += x;
	//m_boneMinLimits += x;

	//std::cout << "init x = " << x.transpose() << std::endl;
	Vector3 vgoal = goal;
	OptimizeFunctor functor(*this, vgoal);


	typedef OptimizeFunctor DfFunctor;
	//typedef Eigen::NumericalDiff<OptimizeFunctor> DfFunctor;

	//Eigen::MatrixXf aJac(functor.values(), functor.inputs());
	//Eigen::MatrixXf nJac(functor.values(), functor.inputs());
	//Eigen::VectorXf ep(functor.values());

	//Eigen::NumericalDiff<OptimizeFunctor> ndffunctor(functor);
	//functor(x, ep);
	//functor.df(x, aJac);
	//ndffunctor.df(x, nJac);

	Eigen::LevenbergMarquardt<DfFunctor, float> lm(functor);
	lm.parameters.maxfev = m_maxItrs;
	lm.parameters.xtol = m_tol;
	lm.parameters.ftol = m_tol;
	lm.parameters.gtol = m_tol;

	auto code = lm.minimize(x);
	std::cout << "iteration = " << lm.iter << std::endl;
	std::cout << "ret = " << code << std::endl;
	std::cout << "x = " << x.transpose() << std::endl;

	//lm.minimizeInit(x);

	//if (code != LevenbergMarquardtSpace::Status::CosinusTooSmall)

	DecodeRotations(x, rotations);

	return true;
}

bool XM_CALLCONV ChainInverseKinematics::solveWithStyle(FXMVECTOR goal, array_view<Quaternion> rotations, array_view<Quaternion> styleReference, float styleReferenceWeight) const
{
	auto n = m_bones.size();
	Eigen::VectorXf x(n * 3);
	Eigen::VectorXf ref(n * 3);
	EncodeRotations(x, rotations);
	EncodeRotations(ref, styleReference);

	OptimizeFunctor functor(*this, goal);
	functor.setReference(ref, styleReferenceWeight);

	Eigen::LevenbergMarquardt<OptimizeFunctor, float> lm(functor);
	lm.parameters.maxfev = m_maxItrs;
	lm.parameters.xtol = m_tol;
	lm.parameters.ftol = m_tol;
	lm.parameters.gtol = m_tol;

	auto code = lm.minimize(x);

	DecodeRotations(x, rotations);

	return true;
}
