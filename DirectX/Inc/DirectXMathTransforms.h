#pragma once
#include "DirectXMathSimpleVectors.h"

namespace DirectX
{
	template <class Derived>
	struct TransformBase : public DirectX::AlignedNew<XMVECTOR>
	{
		typedef Derived DerivedType;

		inline Derived& Inverse()
		{
			return Derived::Inverse();
		}

		inline Derived Inversed() const
		{
			Derived t = static_cast<const Derived&>(*this);
			t.Inverse();
			return t;
		}

		inline XMMATRIX TransformMatrix() const
		{
			return Derived::TransformMatrix();
		}

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX M)
		{
			Derived::SetFromTransformMatrix(M);
		}

		// transform concate
		template <typename Derived>
		inline Derived& operator *=(const TransformBase<Derived>& transform)
		{
			XMMATRIX mat = TransformMatrix();
			XMMATRIX mat2 = transform.TransformMatrix();
			mat *= mat2;
			SetFromTransformMatrix(mat);
			return *this;
		}
	};

	template <class Derived>
	inline Vector3 operator*(const Vector3& v, const TransformBase<Derived>& transform)
	{
		XMVECTOR V = XMLoad(v);
		V = XMVector3TransformCoord(V, transform.TransformMatrix());
		return Vector3(V);
	}

	XM_ALIGNATTR
	struct RotationTransform : public TransformBase<RotationTransform>
	{
		XM_ALIGNATTR
			Quaternion  Rotation;
		// Extract the Matrix Representation of this rigid transform
		inline XMMATRIX TransformMatrix() const
		{
			XMMATRIX M = XMMatrixRotationQuaternion(XMLoadFloat4A(reinterpret_cast<const XMFLOAT4A*>(&Rotation)));
			return M;
		}

		inline RotationTransform() = default;

		inline RotationTransform(const Quaternion& q)
			: Rotation(q)
		{
		}

		inline operator XMVECTOR() const
		{
			return XMLoadFloat4A(reinterpret_cast<const XMFLOAT4A*>(&Rotation));
		}

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX transform)
		{
			XMVECTOR scl, rot, tra;
			XMMatrixDecompose(&scl, &rot, &tra, transform);
			//XMLoadFloat4A(reinterpret_cast<const XMFLOAT4A*>(&Rotation))
			XMStoreFloat4A(reinterpret_cast<XMFLOAT4A*>(&Rotation), rot);
			//Rotation.StoreA(rot);
		}
	};

	inline Vector3 operator*(const Vector3& v, const RotationTransform& transform)
	{
		XMVECTOR V = XMLoad(v);
		XMVECTOR Q = XMLoad(transform.Rotation);
		V = XMVector3Rotate(V, Q);
		return Vector3(V);
	}

	XM_ALIGNATTR
	struct TranslationTransform : public TransformBase<TranslationTransform>
	{
		XM_ALIGNATTR
			Vector3  Translation;
		float Tw;
		// Extract the Matrix Representation of this rigid transform
		inline XMMATRIX TransformMatrix() const
		{
			XMMATRIX M = XMMatrixTranslationFromVector(XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(&Translation)));
			return M;
		}

		inline operator XMVECTOR() const
		{
			return XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(&Translation));
		}

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX transform)
		{
			XMVECTOR scl, rot, tra;
			XMMatrixDecompose(&scl, &rot, &tra, transform);
			XMStoreFloat3A(reinterpret_cast<XMFLOAT3A*>(&Translation), tra);
			//Translation.StoreA(tra);
		}
	};

	inline Vector3 operator*(const Vector3& v, const TranslationTransform& transform)
	{
		return v + transform.Translation;
	}

	XM_ALIGNATTR
	struct ScaleTransform : public TransformBase<ScaleTransform>
	{
		XM_ALIGNATTR
			Vector3 Scale;
		float Sw; // Padding		
				  // Extract the Matrix Representation of this rigid transform
		inline XMMATRIX TransformMatrix() const
		{
			XMMATRIX M = XMMatrixScalingFromVector(XMLoadA(Scale));
			return M;
		}

		operator XMVECTOR() const
		{
			return XMLoadA(Scale);
		}

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX transform)
		{
			XMVECTOR scl, rot, tra;
			XMMatrixDecompose(&scl, &rot, &tra, transform);
			XMStoreA(Scale, scl);
		}
	};

	inline Vector3 operator*(const Vector3& v, const ScaleTransform& transform)
	{
		return XMLoad(v) * XMLoadA(transform.Scale);
	}

	XM_ALIGNATTR
		// Composition of Translation and Rotation
	struct RigidTransform : public TransformBase<RigidTransform>
	{
	public:
		XM_ALIGNATTR
			Quaternion  Rotation;
		XM_ALIGNATTR
			Vector3		Translation;
		float		Tw; // padding

		RigidTransform()
			: Tw(1.0f)
		{
		}

		// Extract the Matrix Representation of this rigid transform
		inline XMMATRIX TransformMatrix() const
		{
			XMMATRIX M = XMMatrixRotationQuaternion(Rotation);
			XMVECTOR VTranslation = XMVectorSelect(g_XMSelect1110.v, Translation, g_XMSelect1110.v);
			M.r[3] = XMVectorAdd(M.r[3], VTranslation);
			return M;
		}

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX transform)
		{
			XMVECTOR scl, rot, tra;
			XMMatrixDecompose(&scl, &rot, &tra, transform);
			Rotation = rot;
			Translation = tra;
		}

		// x = 
		inline void Inverse()
		{
			// R' = R^-1
			XMVECTOR q = XMLoadA(Rotation);
			q = XMQuaternionInverse(q);
			XMStoreA(Rotation, q);

			// T' = -T*R^-1
			XMVECTOR v = XMLoadA(Translation);
			v = -v;
			v = XMVector3Rotate(v, q);
			XMStoreA(Translation, v);
		}

		template <typename _TTransform>
		inline RigidTransform& operator *=(const _TTransform& transform)
		{
			XMMATRIX mat = TransformMatrix();
			XMMATRIX mat2 = transform.TransformMatrix();
			mat *= mat2;
			SetFromTransformMatrix(mat);
			return *this;
		}

		// Rigid * Rigid
		template <>
		inline RigidTransform& operator *=(const RigidTransform& global)
		{
			auto& local = *this;
			XMVECTOR ParQ = XMLoadA(global.Rotation);
			XMVECTOR Q = XMLoadA(local.Rotation);
			Q = XMQuaternionMultiply(Q, ParQ);
			XMStoreA(this->Rotation, Q);

			XMVECTOR V = XMLoadA(local.Translation);
			V = XMVector3Rotate(V, ParQ);
			XMVECTOR gV = XMLoadA(global.Translation);
			V = XMVectorAdd(V, gV);

			XMStoreA(this->Translation, V);
			return *this;
		}

		template <>
		inline RigidTransform& operator *=(const Quaternion& rot)
		{
			auto& local = *this;
			XMVECTOR ParQ = XMLoad(rot);
			XMVECTOR Q = XMQuaternionMultiply(XMLoadA(local.Rotation), ParQ);
			XMStoreA(this->Rotation, Q);

			XMVECTOR V = XMLoadA(local.Translation);
			V = XMVector3Rotate(V, ParQ);

			XMStoreA(this->Translation, V);
			return *this;
		}

		template <>
		inline RigidTransform& operator *=(const Vector3& trans)
		{
			this->Translation += trans;
			return *this;
		}

		// Extract the Dual-Quaternion Representation of this rigid transform
		inline XMDUALVECTOR TransformDualQuaternion() const
		{
			return XMDualQuaternionRotationTranslation(Rotation, Translation);
		}

		bool NearEqual(const RigidTransform& rhs, float tEpsilon = 0.002f, float rEpsilon = 0.5f) const
		{
			Vector3 PosDiff = Translation - rhs.Translation;
			XMVECTOR RotDiff = Rotation;
			RotDiff = XMQuaternionInverse(RotDiff);
			RotDiff = XMQuaternionMultiply(RotDiff, rhs.Rotation);
			float AngDiff = 2 * acosf(XMVectorGetW(RotDiff));
			return (PosDiff.LengthSquared() <= tEpsilon*tEpsilon && AngDiff <= rEpsilon);
		}
	};

	inline Vector3 operator*(const Vector3& v, const RigidTransform& transform)
	{
		XMVECTOR V = XMLoad(v);
		XMVECTOR Q = XMLoadA(transform.Rotation);
		V = XMVector3Rotate(V, Q);
		Q = XMLoadA(transform.Translation);
		V += Q;
		return Vector3(V);
	}

	XM_ALIGNATTR
		// (Scale*)-Rotation-Translation
		//! SRT(RST) transform is not a 'Group' in term of MATRIX production
		//! Thus, We define SRT * SRT -> (S0*S1)*R0*T0*R1*T1
		//! A simple extension to rigid transform, but will apply well for uniform scaling
	struct IsometricTransform : public TransformBase<IsometricTransform>
	{
	public:
		XM_ALIGNATTR
			Quaternion  Rotation;
		XM_ALIGNATTR
			Vector3		Translation;
		float			Tw; // padding
		XM_ALIGNATTR
			Vector3		Scale;
		float			Sw; // Padding

		static IsometricTransform Identity()
		{
			return IsometricTransform();
		}


		IsometricTransform()
			: Scale(1.0f)
		{}

		inline explicit IsometricTransform(CXMMATRIX transform)
		{
			SetFromTransformMatrix(transform);
		}

		operator RigidTransform& ()
		{
			return reinterpret_cast<RigidTransform&>(*this);
		}

		operator const RigidTransform& () const
		{
			return reinterpret_cast<const RigidTransform&>(*this);
		}

		inline explicit IsometricTransform(const RigidTransform &rigid)
			: Rotation(rigid.Rotation), Translation(rigid.Translation), Scale(1.0f)
		{
		}

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX transform)
		{
			XMVECTOR scl, rot, tra;
			XMMatrixDecompose(&scl, &rot, &tra, transform);
			Scale = scl;
			Rotation = rot;
			Translation = tra;
		}

		// x = 
		inline void Inverse()
		{
			// S' = 1 / S;
			XMVECTOR s = XMVectorReciprocal(XMLoadA(Scale));
			XMStoreA(Scale, s);

			// R' = R^-1
			XMVECTOR q = XMLoadA(Rotation);
			q = XMQuaternionConjugate(q);
			XMStoreA(Rotation, q);

			// T' = -T*R^-1 * S^-1
			q = XMVector3Rotate(XMLoadA(Translation), q);
			q = -q;
			q = q * s;
			XMStoreA(Translation, q);
		}


		// transform concate
		template <typename Derived>
		inline IsometricTransform& operator *=(const TransformBase<Derived>& transform)
		{
			XMMATRIX mat = TransformMatrix();
			XMMATRIX mat2 = transform.TransformMatrix();
			mat *= mat2;
			SetFromTransformMatrix(mat);
			return *this;
		}
		// SRT * SRT -> (S0*S1)*R0*T0*R1*T1
		// caculate the transform of Local transform 'this' conacting with Global transform 'rhs'
		template <>
		inline IsometricTransform& operator *=(const TransformBase<IsometricTransform>& glb)
		{
			auto& local = *this;
			auto& global = static_cast<const IsometricTransform&>(glb);
			XMVECTOR ParQ = XMLoadA(global.Rotation);
			XMVECTOR Q = XMQuaternionMultiply(XMLoadA(local.Rotation), ParQ);
			XMVECTOR ParS = XMLoadA(global.Scale);
			XMVECTOR S = ParS*XMLoadA(local.Scale);
			XMStoreA(this->Rotation, Q);
			XMStoreA(this->Scale, S);

			XMVECTOR V = XMLoadA(local.Translation);
			V *= ParS;
			V = XMVector3Rotate(V, ParQ);
			V = XMVectorAdd(V, XMLoadA(global.Translation));

			XMStoreA(this->Translation, V);
			return *this;
		}

		template <>
		IsometricTransform& operator *=(const TransformBase<ScaleTransform>& glb)
		{
			auto& local = *this;
			auto& global = static_cast<const ScaleTransform&>(glb);
			XMVECTOR ParS = XMLoadA(global.Scale);
			XMVECTOR S = ParS * XMLoadA(local.Scale);
			XMStoreA(this->Scale, S);

			XMVECTOR V = XMLoadA(local.Translation);
			V *= ParS;
			XMStoreA(this->Translation, V);
			return *this;
		}


		// ScaledRigid * Rigid
		template <>
		IsometricTransform& operator *=(const TransformBase<RigidTransform>& glb)
		{
			auto& local = *this;
			auto& global = static_cast<const RigidTransform&>(glb);
			XMVECTOR ParQ = XMLoadA(global.Rotation);
			XMVECTOR Q = XMQuaternionMultiply(XMLoadA(local.Rotation), ParQ);
			XMStoreA(this->Rotation, Q);

			XMVECTOR V = XMLoadA(local.Translation);
			V = XMVector3Rotate(V, ParQ);
			V = XMVectorAdd(V, XMLoadA(global.Translation));

			XMStoreA(this->Translation, V);
			return *this;
		}

		IsometricTransform& operator *=(const Quaternion& globalRot)
		{
			auto& local = *this;
			XMVECTOR ParQ = XMLoad(globalRot);
			XMVECTOR Q = XMQuaternionMultiply(XMLoadA(local.Rotation), ParQ);
			XMStoreA(this->Rotation, Q);

			XMVECTOR V = XMLoadA(local.Translation);
			V = XMVector3Rotate(V, ParQ);

			XMStoreA(this->Translation, V);
			return *this;
		}

		template <>
		IsometricTransform& operator *=(const TransformBase<RotationTransform>& glb)
		{
			auto& local = *this;
			auto& global = static_cast<const RotationTransform&>(glb);
			XMVECTOR ParQ = XMLoadA(global.Rotation);
			XMVECTOR Q = XMQuaternionMultiply(XMLoadA(local.Rotation), ParQ);
			XMStoreA(this->Rotation, Q);

			XMVECTOR V = XMLoadA(local.Translation);
			V = XMVector3Rotate(V, ParQ);

			XMStoreA(this->Translation, V);
			return *this;
		}

		template <>
		IsometricTransform& operator *=(const TransformBase<TranslationTransform>& glb)
		{
			auto& global = static_cast<const TranslationTransform&>(glb);
			this->Translation += global.Translation;
			return *this;
		}

		inline XMMATRIX TransformMatrix() const
		{
			XMMATRIX M = XMMatrixScalingFromVector(Scale);
			M *= XMMatrixRotationQuaternion(Rotation);
			XMVECTOR VTranslation = XMVectorSelect(g_XMSelect1110.v, Translation, g_XMSelect1110.v);
			M.r[3] = XMVectorAdd(M.r[3], VTranslation);
			return M;
		}

		explicit operator XMMATRIX() const
		{
			return TransformMatrix();
		}

		static void Lerp(_Out_ IsometricTransform& out, _In_ const IsometricTransform& t0, _In_ const IsometricTransform& t1, float t)
		{
			XMVECTOR tv = XMVectorReplicate(t);
			out.Scale = XMVectorLerpV(XMLoadA(t0.Scale), XMLoadA(t1.Scale), tv);
			out.Translation = XMVectorLerpV(XMLoadA(t0.Translation), XMLoadA(t1.Translation), tv);
			out.Rotation = XMQuaternionSlerpV(XMLoadA(t0.Rotation), XMLoadA(t1.Rotation), tv);
		}

		static void LerpV(_Out_ IsometricTransform& out, _In_ const IsometricTransform& t0, _In_ const IsometricTransform& t1, FXMVECTOR tv)
		{
			out.Scale = XMVectorLerpV(XMLoadA(t0.Scale), XMLoadA(t1.Scale), tv);
			out.Translation = XMVectorLerpV(XMLoadA(t0.Translation), XMLoadA(t1.Translation), tv);
			out.Rotation = XMQuaternionSlerpV(XMLoadA(t0.Rotation), XMLoadA(t1.Rotation), tv);
		}
	};


	inline Vector3 operator*(const Vector3& v, const IsometricTransform& transform)
	{
		XMVECTOR V = XMLoad(v);
		XMVECTOR Q = XMLoadA(transform.Scale);
		V = XMVectorMultiply(V, Q);
		Q = XMLoadA(transform.Rotation);
		V = XMVector3Rotate(V, Q);
		Q = XMLoadA(transform.Translation);
		V = XMVectorAdd(V,Q);
		return Vector3(V);
	}

	XM_ALIGNATTR
	struct LinearTransform : public TransformBase<LinearTransform>, public Matrix4x4
	{
		using Matrix4x4::operator();
		using Matrix4x4::operator+=;
		using Matrix4x4::operator-=;
		using Matrix4x4::operator*=;
		using Matrix4x4::operator/=;
		using Matrix4x4::operator-;
		using Matrix4x4::operator DirectX::XMMATRIX;

		inline void XM_CALLCONV SetFromTransformMatrix(FXMMATRIX transform)
		{
			*this = transform;
		}

		LinearTransform& operator=(const Matrix4x4& rhs)
		{
			Matrix4x4::operator=(rhs);
		}

		template <typename _TTransform>
		LinearTransform& operator *=(const _TTransform& transform)
		{
			XMMATRIX mat = TransformMatrix();
			XMMATRIX mat2 = transform.TransformMatrix();
			mat *= mat2;
			SetFromTransformMatrix(mat);
			return *this;
		}

		inline XMMATRIX TransformMatrix() const
		{
			return XMLoadFloat4x4A(reinterpret_cast<const XMFLOAT4X4A*>(this));
		}
	};	
}
