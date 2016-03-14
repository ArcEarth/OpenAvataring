#include "pch_bcl.h"
#include "InverseKinematics.h"
#include <Eigen\Core>
#include <unsupported\Eigen\NonLinearOptimization>
#include "Tests.h"

#define _DEBUG_JACCOBI 1

using namespace Causality;

enum RotationEncodeMethodsEnum
{
	EulerAngles = 0,
	LnQuaterternion = 1,
};

static constexpr RotationEncodeMethodsEnum RotationEncodeMethod = LnQuaterternion;

namespace DirectX
{
	// Matrix Representation of Cross Production
	// =============Usage===============
	// This matrix is for row major vectors
	// Vector3Transform(V2,MatrixCrossProduct(V1)) == Vector3Cross(V1,V2)
	// Vector3Transform(V2,MatrixTranspose(MatrixCrossProduct(V1))) == Vector3Cross(V2,V1)
	// =============Internal============
	// \mathbf{a} \times \mathbf{b} = [\mathbf{a}]_{\times} \mathbf{b} = \begin{bmatrix}\,0&\!-a_3&\,\,a_2\\ \,\,a_3&0&\!-a_1\\-a_2&\,\,a_1&\,0\end{bmatrix}\begin{bmatrix}b_1\\b_2\\b_3\end{bmatrix}
	// \mathbf{ a } \times \mathbf{ b } = [\mathbf{ b }]_{ \times }^\mathrm T \mathbf{ a } = \begin{ bmatrix }\, 0 & \, \, b_3&\!- b_2\\ - b_3 & 0 & \, \, b_1\\\, \, b_2&\!- b_1&\, 0\end{ bmatrix }\begin{ bmatrix }a_1\\a_2\\a_3\end{ bmatrix }
	// [\mathbf{a}]_{\times} \stackrel{\rm def}{=} \begin{bmatrix}\,\,0&\!-a_3&\,\,\,a_2\\\,\,\,a_3&0&\!-a_1\\\!-a_2&\,\,a_1&\,\,0\end{bmatrix}.
	inline XMMATRIX XM_CALLCONV XMMatrixCrossProduct(FXMVECTOR V)
	{
		// negate v
		XMVECTOR v = XMVectorSelect(g_XMSelect1110.v, V, g_XMSelect1110.v);
		XMVECTOR nv = XMVectorNegate(v);
		XMMATRIX m;
		m.r[0] = _DXMEXT XMVectorPermute<3, 2, 1 + 4, 3 + 4>(v, nv); // [0, a3, -a2]
		m.r[1] = _DXMEXT XMVectorPermute<2 + 4, 3 + 4, 0, 3>(v, nv);	 // [-a3, 0, a1]
		m.r[2] = _DXMEXT XMVectorPermute<1, 0 + 4, 3, 3>(v, nv);		 // [a2,-a1, 0 ]
		m.r[3] = XMVectorZero();
		return m;
	}

	bool XMMatrixCrossProductTest()
	{
		XMVECTOR v1 = XMVectorSet(0, 1, 0, 0);
		v1 = XMVector3Rotate(v1, XMQuaternionRotationRollPitchYaw(0.01, 0.01, 0.5));

		XMVECTOR v2 = XMVectorSet(0, 0.01, 0, 0);
		XMVECTOR right = XMVector3Cross(v1, v2);
		
		XMMATRIX m1 = XMMatrixCrossProduct(v1);
		XMVECTOR result = XMVector3TransformNormal(v2, m1);

		return XMVector3NearEqual(result, right, XMVectorReplicate(0.001f));
	}
}

namespace Test
{
	using namespace DirectX;
	using namespace std;

	ostream& operator<< (ostream& os, const Matrix4x4& m)
	{
		os << m.m[0][0] << ',' << m.m[0][1] << ',' << m.m[0][2] << ',' << m.m[0][3] << endl;
		os << m.m[1][0] << ',' << m.m[1][1] << ',' << m.m[1][2] << ',' << m.m[1][3] << endl;
		os << m.m[2][0] << ',' << m.m[2][1] << ',' << m.m[2][2] << ',' << m.m[2][3] << endl;
		os << m.m[3][0] << ',' << m.m[3][1] << ',' << m.m[3][2] << ',' << m.m[3][3] << endl;
		return os;
	};

	void QuaternionMultiplyTest(const Quaternion &q, const Vector3 &v, const Quaternion &ldq);

	float randf()
	{
		float r = (float)rand();
		r /= (float)(RAND_MAX);
		return r;
	}

	float rand2pi()
	{
		return randf() * XM_2PI - XM_PI;
	}

	float randpi()
	{
		return randf() * XM_PI - XM_PIDIV2;
	}

	bool QuaternionEulerTest()
	{
		XMMatrixCrossProductTest();
		// Patch Yaw Roll = (1.4,0.8,0.4)
		Vector3 v(0, 1, 0);
		Quaternion ldq;
		Quaternion q = XMQuaternionRotationRollPitchYaw(0.01, 0.01, 0.5);
		//Quaternion ldq(0.001, 0.0, 0.0, 0);
		//QuaternionMultiplyTest(q, v, ldq);
		//cout << "=====" << endl;

		ldq = Quaternion(0.0, 0.001, 0.0, 0);
		QuaternionMultiplyTest(q, v, ldq);
		cout << "=====" << endl;

		v = XMVector3Rotate(v, q);
		q = XMQuaternionIdentity();

		XMVECTOR v0 = XMVector3Rotate(v, XMVectorSet(-0.000244786148,0.000958836172,6.69062138e-06,0.999999464));
		XMVECTOR v1 = XMVector3Rotate(v, XMVectorSet(0.000000000,0.000999999815,0.000000000,0.999999523));
		v0 -= v;
		v1 -= v;

		ldq = Quaternion(0.001, 0.0, 0.0, 0);
		QuaternionMultiplyTest(q, v, ldq);
		cout << "=====" << endl;
		ldq = Quaternion(0.0, 0.001, 0.0, 0);
		QuaternionMultiplyTest(q, v, ldq);
		cout << "=====" << endl;
		return true;
	}

	void QuaternionMultiplyTest(const Quaternion &q, const Vector3 &v, const Quaternion &ldq)
	{
		XMVECTOR axis;
		float ang;
		XMQuaternionToAxisAngle(&axis, &ang, q);
		axis = XMVector3Normalize(axis);
		ang *= 0.5;

		cout << "q = " << q << endl;

		Quaternion lq = XMQuaternionLn(q);
		cout << "ln(q) = " << lq << endl;

		cout << "dlnq = d(ln(q)) = " << ldq << endl;
		Quaternion dq = XMQuaternionExp(ldq);
		cout << "dq = exp(d(ln(q))) = " << dq << endl;

		dq = XMQuaternionExp((lq + ldq));
		cout << "exp(ln(q)+d(ln(q))) = " << dq << endl;
		dq = XMQuaternionMultiply(XMQuaternionConjugate(q), dq);
		cout << "dq = q^-1 * exp(ln(q)+d(ln(q))) = " << dq << endl;

		cout << "||dq|| =" << XMVectorGetX(XMVector3Length(dq)) << endl;

		float sinadiva = 1.0f;
		if (fabs(ang) > std::numeric_limits<float>::epsilon())
			sinadiva = sin(ang) / ang;

		Quaternion estimate = sinadiva * (cos(ang) * (XMVECTOR)ldq + sin(ang)*XMVector3Cross(axis, ldq));
		Quaternion another = XMVector3Dot(ldq, axis);
		//another *= axis;
		//estimate += another;

		cout << "estimate = " << estimate << endl;

		XMVECTOR cosA = XMVectorSplatW(q);
		XMVECTOR sinA = XMVectorSqrt(g_XMOne.v - cosA * cosA);

		//Matrix4x4 jac = XMVectorGetX(sinA) * XMMatrixCrossProduct(axis) + XMMatrixScalingFromVector(cosA);

		Matrix4x4 jac = sinadiva* (sin(ang)* XMMatrixCrossProduct(axis) + cos(ang) * XMMatrixIdentity());
		cout << "jaccobi == " << endl << jac << endl;
		Quaternion estimate2 = XMVector3TransformNormal(ldq, jac);
		cout << "estimate2 = " << estimate2 << endl;

		Vector3 qv = XMVector3Rotate(v, q);
		cout << "qv = " << qv << endl;

		Matrix4x4 derv = -XMMatrixCrossProduct(qv);
		cout << endl << "analatic derv d(qv)/d(q) == " << endl << derv;

		derv = jac * derv;
		cout << endl << "overall derv == " << endl << derv ;

		Vector3 derest = Vector3::TransformNormal(Vector3(ldq), derv);
		derest *= XMVector3ReciprocalLength(ldq) * 2;
		cout << "new analatic derv = " << derest << endl;


		Vector3 dotvdq = XMVector3Dot(qv, dq);
		Vector3 crossvdq = XMVector3Cross(qv, dq);
		cout << "dot(qv,dq) = " << dotvdq << endl;
		cout << "cross(qv) = " << crossvdq << endl;

		Vector3 dif = XMVector3Rotate(qv, dq) - (XMVECTOR)qv;
		dif *= XMVector3ReciprocalLength(dq);

		cout << "d(rv)/dq = " << dif << endl;

		lq = XMQuaternionMultiply(q, dq);
		cout << "q * dq= " << lq << endl;

		lq = XMQuaternionMultiply(dq, q);
		cout << "dq * q= " << lq << endl;
	}

	bool InverseKinematicsTest()
	{
		QuaternionEulerTest();
		using namespace DirectX;

		ChainInverseKinematics cik(3);
		cik.bone(0) = Vector4(0, 1.0f, 0, 1.0f);
		cik.bone(1) = Vector4(0, 1.0f, 0, 1.0f);
		cik.bone(2) = Vector4(0, 1.0f, 0, 1.0f);
		//cik.m_boneMinLimits[0].x = 0;
		//cik.m_boneMaxLimits[0].x = 0;
		cik.minRotation(1).y = 0;
		cik.maxRotation(1).y = 0;
		cik.minRotation(2).y = 0;
		cik.maxRotation(2).y = 0;
		//cik.computeJointWeights();
		vector<Quaternion> rotations(3);
		for (size_t i = 0; i < 1; i++)
		{
			rotations[0] = XMQuaternionRotationRollPitchYaw(0.01, 0, 0.01);
			rotations[1] = XMQuaternionRotationRollPitchYaw(0.01, 0, 0.5);
			rotations[2] = XMQuaternionRotationRollPitchYaw(0.5, 0.01, 0.5);
			Vector3 goal = XMVectorSet(1.0f + randf(), randf(), randf(), 0);
			auto code = cik.solve(goal, rotations);
			Vector3 achieved = cik.endPosition(rotations);
			cout << "ik test : goal = " << goal << " ; achieved position = " << achieved << endl;
		}
		cout << "rotations = {" << std::endl;
		for (size_t i = 0; i < 3; i++)
		{
			cout << "  " << rotations[i] << std::endl;
		}
		cout << '}' << endl;

		return true;
	}

	REGISTER_TEST_METHOD(InverseKinematicsTest,InverseKinematicsTest);
}

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
	static void DecodeRotationsEuler(const Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> rotations)
	{
		int n = rotations.size();
		for (int i = 0; i < n; i++)
		{
			auto euler = x.segment<3>(i * 3);
			rotations[i] = DirectX::XMQuaternionRotationRollPitchYaw(euler[0], euler[1], euler[2]);
		}
	}

	static void EncodeRotationsEuler(Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> rotations)
	{
		int n = rotations.size();
		for (int i = 0; i < n; i++)
		{
			auto euler = x.segment<3>(i * 3);
			auto dxeuler = DirectX::XMQuaternionEulerAngleYawPitchRoll(rotations[i]);
			DirectX::XMStoreFloat3(euler.data(), dxeuler);
		}
	}

	static void DecodeRotationsLnQ(const Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> rotations)
	{
		int n = rotations.size();
		for (int i = 0; i < n; i++)
		{
			auto lnq = x.segment<3>(i * 3);
			rotations[i] = DirectX::XMQuaternionExp(DirectX::XMLoadFloat3(lnq.data()));
		}
	}

	static void EncodeRotationsLnQ(Eigen::VectorXf &x, Causality::array_view<const DirectX::SimpleMath::Quaternion> rotations)
	{
		int n = rotations.size();
		for (int i = 0; i < n; i++)
		{
			auto lnq = x.segment<3>(i * 3);
			auto dxlnq = DirectX::XMQuaternionLn(rotations[i]);
			DirectX::XMStoreFloat3(lnq.data(), dxlnq);
		}
	}

	static inline void DecodeRotations(const Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> rotations)
	{
		if (RotationEncodeMethod == EulerAngles)
			DecodeRotationsEuler(x, rotations);
		else
			DecodeRotationsLnQ(x, rotations);
	}

	static inline void EncodeRotations(Eigen::VectorXf &x, Causality::array_view<DirectX::SimpleMath::Quaternion> rotations)
	{
		if (RotationEncodeMethod == EulerAngles)
			EncodeRotationsEuler(x, rotations);
		else
			EncodeRotationsLnQ(x, rotations);
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
	mutable std::vector<DirectX::Vector3, DirectX::XMAllocator>	m_jac;

	void fillRotations(const InputType &x) const
	{
		assert(x.size() == 3 * n);
		DecodeRotations(x, m_rots);
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

		assert(fvec.size() == 3 + x.size());

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

		if (RotationEncodeMethod == EulerAngles)
			ik.endPositionJaccobiRespectEuler(m_rots, m_jac);
		else
		{
			ik.endPositionJaccobiRespectLnQuaternion(m_rots, m_jac);
		}

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
	for (int i = m_bones.size() - 1; i >= 0; --i)
	{
		V = XMLoadA(m_bones[i]);
		V = XMVector3Length(V);
		LV += V;
		m_jointWeights[i] = LV;
	}
}

// Jaccobbi from a rotation radius vector (r) respect to a small rotation dr = (drx,dry,drz) in global reference frame
// [\mathbf{a}]_{\times} \stackrel{\rm def}{=} \begin{bmatrix}\,\,0&\!-a_3&\,\,\,a_2\\\,\,\,a_3&0&\!-a_1\\\!-a_2&\,\,a_1&\,\,0\end{bmatrix}.
void ChainInverseKinematics::jacobbiRespectAxisAngle(Matrix4x4 & j, const float * r)
{
	j._11 = 0, j._12 = -r[2], j._13 = r[1];
	j._21 = r[2], j._22 = 0, j._23 = -r[0];
	j._31 = -r[1], j._32 = r[0], j._33 = 0;
}

// Roll-Patch-Yaw
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
	Q = XMQuaternionMultiply(lQ, Q); // Q = base * roll
	V = XMVectorSet(-r.y, r.x ,0 ,0);
	V = XMVector3Rotate(V, Q);
	MJ.r[2] = V;
	r = XMVector3Rotate(r, lQ);

	lQ = XMQuaternionRotationPatch(euler.x);
	Q = XMQuaternionMultiply(lQ, Q); // Q = base * roll * patch
	V = XMVectorSet(0,-r.z,r.y,0);
	V = XMVector3Rotate(V, Q);
	MJ.r[0] = V;
	r = XMVector3Rotate(r, lQ);

	lQ = XMQuaternionRotationYaw(euler.y);
	Q = XMQuaternionMultiply(lQ, Q); // Q = base * roll * patch * yaw
	V = XMVectorSet(r.z,0,-r.x,0);
	V = XMVector3Rotate(V, Q);
	MJ.r[1] = V;

	//MJ = XMMatrixTranspose(MJ);
	return MJ;
}

XMMATRIX XM_CALLCONV Causality::ChainInverseKinematics::jacobbiTransposeRespectAxisAngle(const Vector3 & rv, FXMVECTOR qrot, FXMVECTOR globalRot)
{
	using namespace DirectX;

	XMVECTOR V;
	XMMATRIX MJ;
	XM_ALIGNATTR Matrix4x4 jac;
	XM_ALIGNATTR Vector4 r;
	V = XMVector3Rotate(rv, qrot);
	XMStoreA(r, V);
	XMVECTOR Q = globalRot; //XMQuaternionMultiply(qrot, globalRot);

	//r.v = rv;
	//XMVECTOR Q = XMQuaternionMultiply(qrot, globalRot);

	jacobbiRespectAxisAngle(jac, &r.x);


	// Rotate each row of the matrix
	V = XMLoadFloat4A(jac.m[0]);
	V = XMVector3Rotate(V, Q);
	MJ.r[0] = V;
	V = XMLoadFloat4A(jac.m[1]);
	V = XMVector3Rotate(V, Q);
	MJ.r[1] = V;
	V = XMLoadFloat4A(jac.m[2]);
	V = XMVector3Rotate(V, Q);
	MJ.r[2] = V;
	MJ.r[3] = XMVectorZero();

	XMVECTOR axis = XMVector3Normalize(qrot);
	XMVECTOR cosA = XMVectorSplatW(qrot);
	XMVECTOR sinA = XMVectorSqrt(g_XMOne.v - cosA * cosA);
	XMVECTOR sinc_a = g_XMOne.v - (g_XMOne.v - cosA) / 3.0; // sinc(x) ~= 1 - x^2/6 == 1 - (1-cos(x))/3

	cosA *= sinc_a;
	sinA *= sinc_a;
	XMMATRIX jacq_lnq = XMVectorGetX(sinA) * XMMatrixCrossProduct(axis) + XMMatrixScalingFromVector(cosA);
	//cout << "jaccobi == " << endl << jac << endl;
	MJ = jacq_lnq * MJ;

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

	// Chain Position Vectors
	auto rad = getRadiusVectors(rotations);

	XMVECTOR gq = XMQuaternionIdentity();
	for (int i = 0; i < n; i++)
	{
		XMVECTOR q = XMLoadA(rotations[i]);
		Vector3 eular = XMQuaternionEulerAngleYawPitchRoll(q);

		auto& r = reinterpret_cast<Vector3&>(rad[i]);
		XMMATRIX jac = jacobbiTransposeRespectEuler(r, eular, gq);

		XMStoreFloat3x3(reinterpret_cast<XMFLOAT3X3*>(&jacb[i * 3].x), jac);

		gq = XMQuaternionMultiply(q, gq);
	}

	//return jacb;
}

void ChainInverseKinematics::endPositionJaccobiRespectAxisAngle(array_view<const Quaternion> rotations, array_view<Vector3> jacb) const
{
	using namespace Eigen;
	using namespace DirectX;
	const auto n = m_bones.size();

	// Chain Position Vectors
	auto rad = getRadiusVectors(rotations);

	XMVECTOR gq = XMQuaternionIdentity();
	for (int i = 0; i < n; i++)
	{
		XMVECTOR q = XMLoadA(rotations[i]);

		auto& r = reinterpret_cast<Vector3&>(rad[i]);
		XMMATRIX jac = jacobbiTransposeRespectAxisAngle(r, q, gq);

		XMStoreFloat3x3(reinterpret_cast<XMFLOAT3X3*>(&jacb[i * 3].x), jac);

		gq = XMQuaternionMultiply(q, gq);
	}
}

void ChainInverseKinematics::endPositionJaccobiRespectLnQuaternion(array_view<const Quaternion> rotations, array_view<Vector3> jacb) const
{
	endPositionJaccobiRespectAxisAngle(rotations, jacb);
	// d(lnQ) == 0.5 * d(axis*angle), thus we need to apply this factor back
	for (auto& v : jacb)
		v *= 2.0f;
}

std::vector<Vector4, DirectX::XMAllocator> ChainInverseKinematics::getRadiusVectors(array_view<const Quaternion> &rotations) const
{
	using namespace DirectX;

	const auto n = m_bones.size();
	std::vector<DirectX::Vector4, DirectX::XMAllocator>	rad(n);

	XMVECTOR q, t, gt;

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

	return rad;
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

	Eigen::MatrixXf aJac(functor.values(), functor.inputs());
	Eigen::MatrixXf nJac(functor.values(), functor.inputs());
	Eigen::VectorXf ep(functor.values());

#if defined(_DEBUG_JACCOBI) && _DEBUG_JACCOBI
	Eigen::NumericalDiff<OptimizeFunctor> ndffunctor(functor);
	functor(x, ep);
	functor.df(x, aJac);
	ndffunctor.df(x, nJac);

	std::cout << "Numberic Jaccobi : " << std::endl << nJac.topRows(3) << std::endl;
	std::cout << "Analatic Jaccobi : " << std::endl << aJac.topRows(3) << std::endl;
#endif

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
