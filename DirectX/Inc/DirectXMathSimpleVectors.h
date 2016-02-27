#pragma once
// Disable Viewport class to prevent <SimpleMath.h> to drag d3d11 headers in
#define _SIMPLE_MATH_NO_VIEWPORT 1
#include "DirectXMathExtend.h"
#include <SimpleMath.h>

namespace DirectX
{
	// SimpleMath::DualQuaternion
	namespace SimpleMath
	{
		class DualQuaternion
		{
		public:
			Quaternion Qr, Qe;
			DualQuaternion()
				: Qr(), Qe(0.0f, 0.0f, 0.0f, 0.0f)
			{}
			DualQuaternion(const Quaternion& Rotation, const Vector3& Translation)
				: Qr(Rotation)
			{
				Qr.Normalize();
				const float* Q0 = reinterpret_cast<float*>(&Qr);
				const float* T = reinterpret_cast<const float*>(&Translation);
				Qe.w = -0.5f*(T[0] * Q0[0] + T[1] * Q0[1] + T[2] * Q0[2]);
				Qe.x = 0.5f*(T[0] * Q0[3] + T[1] * Q0[2] - T[2] * Q0[1]);
				Qe.y = 0.5f*(-T[0] * Q0[2] + T[1] * Q0[3] + T[2] * Q0[0]);
				Qe.z = 0.5f*(T[0] * Q0[1] - T[1] * Q0[0] + T[2] * Q0[3]);
			}
			DualQuaternion(FXMVECTOR Qr, FXMVECTOR Qe)
				: Qr(Qr), Qe(Qe)
			{}
			DualQuaternion(CXMDUALVECTOR DQ)
				: Qr(DQ.r[0]), Qe(DQ.r[1])
			{}

			explicit DualQuaternion(_In_reads_(8) const float* pArray)
				: Qr(pArray), Qe(pArray + 4)
			{}
			explicit DualQuaternion(_In_reads_(2) const Quaternion* pQArray)
				: Qr(*pQArray), Qe(*(pQArray + 1))
			{}

			inline operator XMDUALVECTOR() const
			{
				XMDUALVECTOR dqRes;
				dqRes.r[0] = Qr;
				dqRes.r[1] = Qe;
				return dqRes;
			}


			void Normarlize(DualQuaternion& result) const
			{
				XMDUALVECTOR dq = *this;
				dq = XMDualQuaternionNormalize(dq);
				result.Qr = dq.r[0];
				result.Qe = dq.r[1];
			}

			void Normarlize()
			{
				Normarlize(*this);
			}

			void Inverse(DualQuaternion& result) const
			{
				XMDUALVECTOR dq = *this;
				dq = XMDualQuaternionInverse(dq);
				result.Qr = dq.r[0];
				result.Qe = dq.r[1];
			}
			void Inverse()
			{
				Inverse(*this);
			}

			void Conjugate()
			{
				Qr.Conjugate();
				Qe.Conjugate();
			}

			void Conjugate(DualQuaternion& result) const
			{
				result.Qr = XMQuaternionConjugate(Qr);
				result.Qe = XMQuaternionConjugate(Qe);
			}

			Vector2 Norm() const
			{
				Vector2 value;
				XMVECTOR q0 = Qr;
				XMVECTOR q1 = Qe;
				XMVECTOR len = XMQuaternionLength(q0);
				q1 = XMVector4Dot(q0, q1);
				q0 = XMVectorDivide(q1, len);
				q1 = XMVectorSelect(len, q0, g_XMSelect0101);
				value = q1;
				return value;
			}

			bool IsUnit() const
			{
				XMVECTOR q0 = Qr;
				XMVECTOR q1 = Qe;
				q1 = XMVector4Dot(q0, q1);
				return XMVector4NearEqual(q0, g_XMZero.v, g_XMEpsilon.v);
			}

			bool Decompose(Quaternion& Rotation, Vector3& Translation) const
			{
				const auto& Q = Rotation = XMQuaternionNormalize(Qr);
				// translation vector:
				Translation.x = 2.0f*(-Qe.w*Q.x + Qe.x*Q.w - Qe.y*Q.z + Qe.z*Q.y);
				Translation.y = 2.0f*(-Qe.w*Q.y + Qe.x*Q.z + Qe.y*Q.w - Qe.z*Q.x);
				Translation.z = 2.0f*(-Qe.w*Q.z - Qe.x*Q.y + Qe.y*Q.x + Qe.z*Q.w);
			}

			DualQuaternion& operator+= (const DualQuaternion& rhs)
			{
				Qr += rhs.Qr;
				Qe += rhs.Qe;
				return *this;
			}

			DualQuaternion& operator-= (const DualQuaternion& rhs)
			{
				Qr -= rhs.Qr;
				Qe -= rhs.Qe;
				return *this;
			}

			DualQuaternion& operator*= (const DualQuaternion& rhs)
			{
				XMVECTOR A = this->Qr;
				XMVECTOR B = this->Qe;
				XMVECTOR C = rhs.Qr;
				XMVECTOR D = rhs.Qe;
				D = XMQuaternionMultiply(A, D);
				B = XMQuaternionMultiply(B, C);
				Qe = XMVectorAdd(D, B);
				Qr = XMQuaternionMultiply(A, C);
			}

			DualQuaternion& operator*= (float scale)
			{
				Qr *= scale;
				Qr *= scale;
				return *this;
			}

			DualQuaternion& operator/= (float scale)
			{
				float s = 1.0f / scale;
				return (*this) *= s;
			}
		};

		inline DualQuaternion operator+ (const DualQuaternion& lhs, const DualQuaternion& rhs)
		{
			DualQuaternion result = lhs;
			result += rhs;
			return result;
		}
		inline DualQuaternion operator- (const DualQuaternion& lhs, const DualQuaternion& rhs)
		{
			DualQuaternion result = lhs;
			result -= rhs;
			return result;
		}
		inline DualQuaternion operator* (const DualQuaternion& lhs, const DualQuaternion& rhs)
		{
			DualQuaternion result = lhs;
			result *= rhs;
			return result;
		}
		inline DualQuaternion operator* (const DualQuaternion& lhs, float rhs)
		{
			DualQuaternion result = lhs;
			result *= rhs;
			return result;
		}
		inline DualQuaternion operator/ (const DualQuaternion& lhs, float rhs)
		{
			DualQuaternion result = lhs;
			result /= rhs;
			return result;
		}
	}

	using SimpleMath::Vector2;
	using SimpleMath::Vector3;
	using SimpleMath::Vector4;

	using SimpleMath::Quaternion;
	using SimpleMath::DualQuaternion;

	using SimpleMath::Color;
	using SimpleMath::Plane;
	using SimpleMath::Ray;

	typedef SimpleMath::Matrix Matrix4x4;

	using SimpleMath::operator*;
	using SimpleMath::operator+;
	using SimpleMath::operator-;
	using SimpleMath::operator/;

	namespace HLSLVectors
	{
		using float4 = Vector4;
		using float3 = Vector3;
		using float2 = Vector2;
		using uint = uint32_t;
		using uint4 = XMUINT4;
		using uint3 = XMUINT3;
		using uint2 = XMUINT2;
		using int2 = XMINT2;
		using int3 = XMINT3;
		using int4 = XMINT4;
		using float4x4 = Matrix4x4;
		using float4x3 = XMFLOAT3X4;
		using float3x3 = XMFLOAT3X3;
		using matrix = float4x4;
	}

	inline XMVECTOR XMLoad(const SimpleMath::Quaternion& src)
	{
		return XMLoadFloat4(&src);
	}
	inline XMVECTOR XMLoadA(const SimpleMath::Quaternion& src)
	{
		return XMLoadFloat4A(reinterpret_cast<const XMFLOAT4A*>(&src));
	}
	inline void XM_CALLCONV XMStore(SimpleMath::Quaternion& dest, FXMVECTOR v)
	{
		XMStoreFloat4(&dest, v);
	}
	inline void XM_CALLCONV XMStoreA(SimpleMath::Quaternion& dest, FXMVECTOR v)
	{
		XMStoreFloat4A(reinterpret_cast<XMFLOAT4A*>(&dest), v);
	}

#ifdef _OSTREAM_
#include <iomanip>
	//inline std::ostream& operator << (std::ostream& lhs, const SimpleMath::Vector3& rhs)
	//{
	//	lhs << '(' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.x
	//		<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.y << ')';
	//	return lhs;
	//};

	//inline std::ostream& operator << (std::ostream& lhs, const SimpleMath::Vector4& rhs)
	//{
	//	lhs << '(' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.x
	//		<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.y
	//		<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.z << ')';
	//	return lhs;
	//};

	//inline std::ostream& operator << (std::ostream& lhs, const SimpleMath::Vector2& rhs)
	//{
	//	lhs << '(' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.x
	//		<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.y
	//		<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.z
	//		<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.w << ')';
	//	return lhs;
	//};

	inline std::ostream& operator << (std::ostream& lhs, const SimpleMath::Quaternion& rhs)
	{
		lhs << (const Vector4&)(rhs);

		float theta = std::acosf(rhs.w) * 2 / DirectX::XM_PI;
		DirectX::Vector3 axis(rhs);
		axis.Normalize();
		lhs << "[axis=(" << axis
			<< "),ang=" << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << theta << "*Pi]";
		return lhs;
	};
#endif

}