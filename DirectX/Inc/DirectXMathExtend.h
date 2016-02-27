#pragma once

#ifndef DX_MATH_EXT_H
#define DX_MATH_EXT_H

#if defined(BT_USE_SSE)
#error Bullet physics can not be include before DirectX Math headers.
#endif

// Disable bullet to overload operators on _m128
#ifndef BT_NO_SIMD_OPERATOR_OVERLOADS
#define BT_NO_SIMD_OPERATOR_OVERLOADS
#endif


#include <DirectXMath.h>
#include <DirectXCollision.h>
#include <DirectXColors.h>
#include <DirectXMathIntrinsics.h>
#ifndef _DXMEXT
#define _DXMEXT
#endif

#ifndef XM_ALIGNATTR
#define XM_ALIGNATTR _MM_ALIGN16
#endif
namespace DirectX
{
	const float XM_EPSILON = 1.192092896e-7f;
	const float XM_INFINITY = 1e12f;

	// vector alignment of XMVECTOR
	const size_t XM_ALIGNMENT = alignof(XMVECTOR);

	XMGLOBALCONST XMVECTORF32 g_XMNegtiveXZ = { -1.0f,1.0f,-1.0f,1.0f };

	inline float XMScalarReciprocalSqrtEst(float x) {
		float xhalf = 0.5f*x;
		int i = *(int*)&x;
		i = 0x5f3759df - (i >> 1);
		x = *(float*)&i;
		x = x*(1.5f - xhalf*x*x);
		return x;
	}

	inline XMMATRIX XMMatrixZero()
	{
		XMMATRIX M;
		M.r[0] = XMVectorZero();
		M.r[1] = XMVectorZero();
		M.r[2] = XMVectorZero();
		M.r[3] = XMVectorZero();
		return M;
	}

	XMVECTOR    XM_CALLCONV     XMVectorAddInt(FXMVECTOR V1, FXMVECTOR V2);
	XMVECTOR    XM_CALLCONV     XMVectorSubtractInt(FXMVECTOR V1, FXMVECTOR V2);
	XMVECTOR    XM_CALLCONV     XMVectorMultiplyInt(FXMVECTOR V1, FXMVECTOR V2);

	inline XMVECTOR XM_CALLCONV XMLoadFloat4(const float* pF4)
	{
		return XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(pF4));
	}
	inline XMVECTOR XM_CALLCONV XMLoadFloat3(const float* pF3)
	{
		return XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(pF3));
	}
	inline XMVECTOR XM_CALLCONV XMLoadFloat4A(const float* pF4A)
	{
		return XMLoadFloat4A(reinterpret_cast<const XMFLOAT4A*>(pF4A));
	}
	inline XMVECTOR XM_CALLCONV XMLoadFloat3A(const float* pF3A)
	{
		return XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(pF3A));
	}
	inline void XM_CALLCONV XMStoreFloat4(float* pF4, FXMVECTOR V)
	{
		return XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(pF4), V);
	}
	inline void XM_CALLCONV XMStoreFloat3(float* pF3, FXMVECTOR V)
	{
		return XMStoreFloat3(reinterpret_cast<XMFLOAT3*>(pF3), V);
	}
	inline void XM_CALLCONV XMStoreFloat4A(float* pF4A, FXMVECTOR V)
	{
		return XMStoreFloat4A(reinterpret_cast<XMFLOAT4A*>(pF4A), V);
	}
	inline void XM_CALLCONV XMStoreFloat3A(float* pF3A, FXMVECTOR V)
	{
		return XMStoreFloat3A(reinterpret_cast<XMFLOAT3A*>(pF3A), V);
	}
	// return V.x + V.y + V.z + V.w
	inline XMVECTOR XM_CALLCONV XMVector4HorizontalSum(FXMVECTOR V)
	{
#if defined(_XM_NO_INTRINSICS_)

		XMVECTOR Result;
		Result.vector4_f32[0] =
			Result.vector4_f32[1] =
			Result.vector4_f32[2] =
			Result.vector4_f32[3] = V.vector4_f32[0] + V.vector4_f32[1] + V.vector4_f32[2] + V.vector4_f32[3];
		return Result;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
		static_assert(false, "ARM Instrinsics not available yet");
		float32x4_t vTemp = vmulq_f32(V1, V2);
		float32x2_t v1 = vget_low_f32(vTemp);
		float32x2_t v2 = vget_high_f32(vTemp);
		v1 = vpadd_f32(v1, v1);
		v2 = vpadd_f32(v2, v2);
		v1 = vadd_f32(v1, v2);
		return vcombine_f32(v1, v1);
#elif defined(_XM_SSE_INTRINSICS_)
		//static_assert(false, "SIDM Instrinsics not tested yet");
		XMVECTOR vT = V;
		vT = _mm_shuffle_ps(vT, V, _MM_SHUFFLE(1, 0, 0, 0)); // Copy X to the Z position and Y to the W position
		vT = _mm_add_ps(vT, V);          // Add Z = X+Z; W = Y+W;
		vT = _mm_shuffle_ps(V, vT, _MM_SHUFFLE(0, 3, 0, 0));  // Copy W to the Z position
		vT = _mm_add_ps(V, vT);           // Add Z and W together
		return XM_PERMUTE_PS(V, _MM_SHUFFLE(2, 2, 2, 2));    // Splat Z and return
#endif
	}

#ifndef DIRECTX_ALLIGNED_NEW
#define DIRECTX_ALLIGNED_NEW
	template<typename T>
	struct AlignedNew
	{
		static const size_t alignment = alignof(T);
		static_assert(alignment > 8, "AlignedNew is only useful for types with > 8 byte alignment. Did you forget a _declspec(align) on TDerived?");
		// Allocate aligned memory.
		static void* operator new (size_t size)
		{

			void* ptr = _aligned_malloc(size, alignment);

		if (!ptr)
			throw std::bad_alloc();

		return ptr;
		}


			// Free aligned memory.
			static void operator delete (void* ptr)
		{
			_aligned_free(ptr);
		}


		// Array overloads.
		static void* operator new[](size_t size)
		{
			return operator new(size);
		}


			static void operator delete[](void* ptr)
		{
			operator delete(ptr);
		}
	};

	template <typename _T, size_t _Align_Boundary = alignof(_T)>
	class AlignedAllocator
	{
	public:
		typedef	_T		 value_type;
		typedef	size_t		 size_type;
		typedef	ptrdiff_t	 difference_type;
		typedef	_T		*pointer;
		typedef const _T		*const_pointer;
		typedef	_T		&reference;
		typedef const _T		&const_reference;
		inline AlignedAllocator() throw() {}

		template <typename _T2>
		inline  AlignedAllocator(const AlignedAllocator<_T2, _Align_Boundary> &) throw() {}
		inline ~AlignedAllocator() throw() {}

		template <size_t _Other_Alignment>
		inline bool operator== (const AlignedAllocator<_T, _Other_Alignment> & rhs) const
		{
			return _Align_Boundary == _Other_Alignment;
		}



		inline pointer adress(reference r)
		{
			return &r;
		}

		inline const_pointer adress(const_reference r) const
		{
			return &r;
		}

		inline pointer allocate(size_type n)
		{
			return (pointer)_mm_malloc(n*sizeof(value_type), _Align_Boundary);
		}

		inline void deallocate(pointer p, size_type)
		{
			_mm_free(p);
		}

		inline void construct(pointer p, const_reference wert)
		{
			::new(p) value_type(wert);
		}

		inline void destroy(pointer p)
		{ /* C4100 */ p; p->~value_type();
		}

		inline size_type max_size() const throw()
		{
			return size_type(-1) / sizeof(value_type);
		}

		template <typename _T2>
		struct rebind { typedef AlignedAllocator<_T2, _Align_Boundary> other; };
	};

	// Aligned Aloocator for holding XMVECTOR/16 btyes aligned data
	typedef AlignedAllocator<XMVECTOR> XMAllocator;

#endif //DIRECTX_ALLIGNED_NEW

#ifdef _MSC_VER
#pragma region XMFLOAT3X4
#endif
	// 3x4 Matrix: 32 bit floating point components
	// Row Major , Usually use to pass float4x3 matrix to GPU
	// Transpose befor store and after load
	struct XMFLOAT3X4
	{
		union
		{
			struct
			{
				float _11, _12, _13, _14;
				float _21, _22, _23, _24;
				float _31, _32, _33, _34;
			};
			float m[3][4];
		};

		XMFLOAT3X4() {}
		XMFLOAT3X4(float m00, float m01, float m02, float m03,
			float m10, float m11, float m12, float m13,
			float m20, float m21, float m22, float m23);
		explicit XMFLOAT3X4(_In_reads_(12) const float *pArray);

		float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
		float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }

		XMFLOAT3X4& operator= (const XMFLOAT3X4& Float4x3);
	};

	inline XMFLOAT3X4::XMFLOAT3X4
		(
			float m00, float m01, float m02, float m03,
			float m10, float m11, float m12, float m13,
			float m20, float m21, float m22, float m23
			)
	{
		m[0][0] = m00;
		m[0][1] = m01;
		m[0][2] = m02;
		m[0][3] = m03;

		m[1][0] = m10;
		m[1][1] = m11;
		m[1][2] = m12;
		m[1][3] = m13;

		m[2][0] = m20;
		m[2][1] = m21;
		m[2][2] = m22;
		m[2][3] = m23;

	}

	//------------------------------------------------------------------------------
	_Use_decl_annotations_
		inline XMFLOAT3X4::XMFLOAT3X4
		(
			const float* pArray
			)
	{
		assert(pArray != nullptr);

		m[0][0] = pArray[0];
		m[0][1] = pArray[1];
		m[0][2] = pArray[2];
		m[0][3] = pArray[3];

		m[1][0] = pArray[4];
		m[1][1] = pArray[5];
		m[1][2] = pArray[6];
		m[1][3] = pArray[7];

		m[2][0] = pArray[8];
		m[2][1] = pArray[9];
		m[2][2] = pArray[10];
		m[2][3] = pArray[11];

	}

	//------------------------------------------------------------------------------
	inline XMFLOAT3X4& XMFLOAT3X4::operator=
		(
			const XMFLOAT3X4& Float4x4
			)
	{
		XMVECTOR V1 = XMLoadFloat4((const XMFLOAT4*)&Float4x4._11);
		XMVECTOR V2 = XMLoadFloat4((const XMFLOAT4*)&Float4x4._21);
		XMVECTOR V3 = XMLoadFloat4((const XMFLOAT4*)&Float4x4._31);

		XMStoreFloat4((XMFLOAT4*)&_11, V1);
		XMStoreFloat4((XMFLOAT4*)&_21, V2);
		XMStoreFloat4((XMFLOAT4*)&_31, V3);

		return *this;
	}

	inline void XMStoreFloat3x4(
		XMFLOAT3X4 *pDestination,
		CXMMATRIX M
		)
	{
		assert(pDestination);
#if defined(_XM_NO_INTRINSICS_)

		pDestination->m[0][0] = M.r[0].vector4_f32[0];
		pDestination->m[0][1] = M.r[0].vector4_f32[1];
		pDestination->m[0][2] = M.r[0].vector4_f32[2];
		pDestination->m[0][3] = M.r[0].vector4_f32[3];

		pDestination->m[1][0] = M.r[1].vector4_f32[0];
		pDestination->m[1][1] = M.r[1].vector4_f32[1];
		pDestination->m[1][2] = M.r[1].vector4_f32[2];
		pDestination->m[1][3] = M.r[1].vector4_f32[3];

		pDestination->m[2][0] = M.r[2].vector4_f32[0];
		pDestination->m[2][1] = M.r[2].vector4_f32[1];
		pDestination->m[2][2] = M.r[2].vector4_f32[2];
		pDestination->m[2][3] = M.r[2].vector4_f32[3];

#elif defined(_XM_ARM_NEON_INTRINSICS_)
		vst1q_f32(reinterpret_cast<float*>(&pDestination->_11), M.r[0]);
		vst1q_f32(reinterpret_cast<float*>(&pDestination->_21), M.r[1]);
		vst1q_f32(reinterpret_cast<float*>(&pDestination->_31), M.r[2]);
#elif defined(_XM_SSE_INTRINSICS_)
		_mm_storeu_ps(&pDestination->_11, M.r[0]);
		_mm_storeu_ps(&pDestination->_21, M.r[1]);
		_mm_storeu_ps(&pDestination->_31, M.r[2]);
#else // _XM_VMX128_INTRINSICS_
#endif // _XM_VMX128_INTRINSICS_
	}

	inline XMMATRIX XMLoadFloat3x4
		(
			const XMFLOAT3X4* pSource
			)
	{
		assert(pSource);
#if defined(_XM_NO_INTRINSICS_)

		XMMATRIX M;
		M.r[0].vector4_f32[0] = pSource->m[0][0];
		M.r[0].vector4_f32[1] = pSource->m[0][1];
		M.r[0].vector4_f32[2] = pSource->m[0][2];
		M.r[0].vector4_f32[3] = pSource->m[0][3];

		M.r[1].vector4_f32[0] = pSource->m[1][0];
		M.r[1].vector4_f32[1] = pSource->m[1][1];
		M.r[1].vector4_f32[2] = pSource->m[1][2];
		M.r[1].vector4_f32[3] = pSource->m[1][3];

		M.r[2].vector4_f32[0] = pSource->m[2][0];
		M.r[2].vector4_f32[1] = pSource->m[2][1];
		M.r[2].vector4_f32[2] = pSource->m[2][2];
		M.r[2].vector4_f32[3] = pSource->m[2][3];

		return M;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
		XMMATRIX M;
		M.r[0] = vld1q_f32(reinterpret_cast<const float*>(&pSource->_11));
		M.r[1] = vld1q_f32(reinterpret_cast<const float*>(&pSource->_21));
		M.r[2] = vld1q_f32(reinterpret_cast<const float*>(&pSource->_31));
		return M;
#elif defined(_XM_SSE_INTRINSICS_)
		XMMATRIX M;
		M.r[0] = _mm_loadu_ps(&pSource->_11);
		M.r[1] = _mm_loadu_ps(&pSource->_21);
		M.r[2] = _mm_loadu_ps(&pSource->_31);
		M.r[3] = g_XMIdentityR3;
		return M;
#elif defined(XM_NO_MISALIGNED_VECTOR_ACCESS)
#endif // _XM_VMX128_INTRINSICS_
	}

#ifdef _MSC_VER
#pragma endregion

#pragma region XMDUALVECTOR
#endif

	struct XMDUALVECTOR;
	// Calling convetion

	// First XMDUALVECTOR
#if ( defined(_M_IX86) || defined(_M_ARM) || defined(_XM_VMX128_INTRINSICS_) || _XM_VECTORCALL_ ) && !defined(_XM_NO_INTRINSICS_)
	typedef const XMDUALVECTOR FXMDUALVECTOR;
#else
	typedef const XMDUALVECTOR& FXMDUALVECTOR;
#endif

	// 2nd
#if ( defined(_M_ARM) || defined(_XM_VMX128_INTRINSICS_) || (_XM_VECTORCALL_)) && !defined(_XM_NO_INTRINSICS_)
	typedef const XMDUALVECTOR GXMDUALVECTOR;
#else
	typedef const XMDUALVECTOR& GXMDUALVECTOR;
#endif
	// 3rd
#if ( defined(_XM_VMX128_INTRINSICS_) || _XM_VECTORCALL_ ) && !defined(_XM_NO_INTRINSICS_)
	typedef const XMDUALVECTOR HXMDUALVECTOR;
#else
	typedef const XMDUALVECTOR& HXMDUALVECTOR;
#endif

#if defined(_XM_VMX128_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
	typedef const XMDUALVECTOR CXMDUALVECTOR;
#else
	typedef const XMDUALVECTOR& CXMDUALVECTOR;
#endif

#if (defined(_M_IX86) || defined(_M_AMD64) || defined(_M_ARM)) && defined(_XM_NO_INTRINSICS_)
	struct XMDUALVECTOR
#else
	__declspec(align(16)) struct XMDUALVECTOR
#endif
	{
#ifdef _XM_NO_INTRINSICS_
		union
		{
			XMVECTOR r[2];
			struct
			{
				float _11, _12, _13, _14;
				float _21, _22, _23, _24;
			};
			float m[2][4];
		};
#else
		XMVECTOR r[2];
#endif

		XMDUALVECTOR() {}
		XMDUALVECTOR(FXMVECTOR V0, FXMVECTOR V1) { r[0] = V0; r[1] = V1; }
		XMDUALVECTOR(float m00, float m01, float m02, float m03,
			float m10, float m11, float m12, float m13)
		{
			r[0] = XMVectorSet(m00, m01, m02, m03);
			r[1] = XMVectorSet(m10, m11, m12, m13);
		}
		explicit XMDUALVECTOR(_In_reads_(8) const float *pArray)
		{
			r[0] = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(pArray));
			r[1] = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(pArray + 4));
		}

		XMDUALVECTOR& XM_CALLCONV operator= (FXMDUALVECTOR DV) { r[0] = DV.r[0]; r[1] = DV.r[1]; return *this; }

		XMDUALVECTOR XM_CALLCONV operator+ () const { return *this; }
		XMDUALVECTOR XM_CALLCONV operator- () const
		{
			XMDUALVECTOR dvResult(r[0], -r[1]);
			return dvResult;
		}

		inline XMVECTOR operator[](int row) const
		{
			assert(row >= 0 && row <= 1);
			return r[row];
		}
		inline XMVECTOR& operator[](int row)
		{
			assert(row >= 0 && row <= 1);
			return r[row];
		}

		XMDUALVECTOR& XM_CALLCONV operator+= (FXMDUALVECTOR M)
		{
			r[0] += M.r[0];
			r[1] += M.r[1];
			return *this;
		}

		XMDUALVECTOR& XM_CALLCONV operator-= (FXMDUALVECTOR M)
		{
			r[0] -= M.r[0];
			r[1] -= M.r[1];
			return *this;
		}

		XMDUALVECTOR&   operator*= (float S)
		{
			r[0] *= S;
			r[1] *= S;
			return *this;
		}

		XMDUALVECTOR&   operator/= (float S)
		{
			r[0] /= S;
			r[1] /= S;
			return *this;
		}
	};

	inline XMVECTOR XM_CALLCONV XMQuaternionSlerpCoefficient(FXMVECTOR t, XMVECTOR xm1)
	{
		const float opmu = 1.90110745351730037f;
		const XMVECTORF32 neg_u0123 = { -1.f / (1 * 3), -1.f / (2 * 5), -1.f / (3 * 7), -1.f / (4 * 9) };
		const XMVECTORF32 neg_u4567 = { -1.f / (5 * 11), -1.f / (6 * 13), -1.f / (7 * 15), -opmu / (8 * 17) };
		const XMVECTORF32 neg_v0123 = { -1.f / 3, -2.f / 5, -3.f / 7, -4.f / 9 };
		const XMVECTORF32 neg_v4567 = { -5.f / 11, -6.f / 13, -7.f / 15, -opmu * 8 / 17 };
		const XMVECTOR one = XMVectorReplicate(1.f);

		XMVECTOR sqrT = XMVectorMultiply(t, t);
		XMVECTOR b0123, b4567, b, c;
		// (b4, b5, b6, b7) = 
		// (x - 1) * (u4 * t^2 - v4, u5 * t^2 - v5, u6 * t^2 - v6, u7 * t^2 - v7) 
		b4567 = _DXMEXT XMVectorNegativeMultiplySubtract(neg_u4567, sqrT, neg_v4567);
		//b4567 = _mm_mul_ps(u4567, sqrT);
		//b4567 = _mm_sub_ps(b4567, v4567);
		b4567 = XMVectorMultiply(b4567, xm1);
		// (b7, b7, b7, b7) 
		b = _DXMEXT XMVectorSwizzle<3, 3, 3, 3>(b4567);
		c = XMVectorAdd(b, one);
		// (b6, b6, b6, b6) 
		b = _DXMEXT XMVectorSwizzle<2, 2, 2, 2>(b4567);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);
		// (b5, b5, b5, b5) 
		b = _DXMEXT XMVectorSwizzle<1, 1, 1, 1>(b4567);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);
		// (b4, b4, b4, b4) 
		b = _DXMEXT XMVectorSwizzle<0, 0, 0, 0>(b4567);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);
		// (b0, b1, b2, b3) = 
		//(x-1)*(u0*t^2-v0,u1*t^2-v1,u2*t^2-v2,u3*t^2-v3)
		b0123 = _DXMEXT XMVectorNegativeMultiplySubtract(neg_u0123, sqrT, neg_v0123);
		//b0123 = _mm_mul_ps(u0123, sqrT);
		//b0123 = _mm_sub_ps(b0123, v0123);
		b0123 = XMVectorMultiply(b0123, xm1);
		// (b3, b3, b3, b3)
		b = _DXMEXT XMVectorSwizzle<3, 3, 3, 3>(b0123);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);
		// (b2, b2, b2, b2)
		b = _DXMEXT XMVectorSwizzle<2, 2, 2, 2>(b0123);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);
		// (b1, b1, b1, b1)
		b = _DXMEXT XMVectorSwizzle<1, 1, 1, 1>(b0123);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);
		// (b0, b0, b0, b0)
		b = _DXMEXT XMVectorSwizzle<0, 0, 0, 0>(b0123);
		c = _DXMEXT XMVectorMultiplyAdd(b, c, one);

		c = XMVectorMultiply(t, c);
		return c;
	}
	// reference
	// Eberly : A Fast and Accurate Algorithm for Computing SLERP
	inline XMVECTOR XM_CALLCONV XMQuaternionSlerpFastV(FXMVECTOR q0, FXMVECTOR q1, FXMVECTOR splatT)
	{
#if defined(_XM_NO_INTRINSICS_)
		// Precomputed constants.
		const float opmu = 1.90110745351730037f;
		const float u[8] = // 1 /[i (2i + 1 )] for i >= 1
		{
			1.f / (1 * 3), 1.f / (2 * 5), 1.f / (3 * 7), 1.f / (4 * 9),
			1.f / (5 * 11), 1.f / (6 * 13), 1.f / (7 * 15), opmu / (8 * 17)
		};
		const float v[8] = // i /(2 i+ 1) for i >= 1
		{
			1.f / 3, 2.f / 5, 3.f / 7, 4.f / 9,
			5.f / 11, 6.f / 13, 7.f / 15, opmu * 8 / 17
		};

		// x = dot(q0,q1) = cos(theta)
		float x = q0.vector4_f32[0] * q1.vector4_f32[0] + q0.vector4_f32[1] * q1.vector4_f32[1] + q0.vector4_f32[2] * q1.vector4_f32[2] + q0.vector4_f32[3] * q1.vector4_f32[3]; // cos (theta)
		float sign = (x >= 0 ? 1 : (x = -x, -1));
		float xm1 = x - 1;
		float d = 1 - t, sqrT = t * t, sqrD = d * d;
		float bT[8], bD[8];
		for (int i = 7; i >= 0; --i)
		{
			bT[i] = (u[i] * sqrT - v[i]) * xm1;
			bD[i] = (u[i] * sqrD - v[i]) * xm1;
		}
		float cT = sign * t *(
			1 + bT[0] * (1 + bT[1] * (1 + bT[2] * (1 + bT[3] * (
				1 + bT[4] * (1 + bT[5] * (1 + bT[6] * (1 + bT[7]))))))));
		float cD = d * (
			1 + bD[0] * (1 + bD[1] * (1 + bD[2] * (1 + bD[3] * (
				1 + bD[4] * (1 + bD[5] * (1 + bD[6] * (1 + bD[7]))))))));
		XMVECTOR slerp = q0 * cD + q1 * cT;
		return slerp;

#elif defined(_XM_SSE_INTRINSICS_) || defined(_XM_ARM_NEON_INTRINSICS_)
		const XMVECTOR signMask = XMVectorReplicate(-0.f);
		const XMVECTOR one = XMVectorReplicate(1.f); // Dot product of 4-tuples. 
		XMVECTOR x = _DXMEXT XMVector4Dot(q0, q1); // cos (theta) in all components
#if defined(_XM_SSE_INTRINSICS_)
		XMVECTOR sign = _mm_and_ps(signMask, x);
		x = _mm_xor_ps(sign, x);
		XMVECTOR localQ1 = _mm_xor_ps(sign, q1);
#else
		uint32x4_t sign = vandq_u32(signMask, x);
		x = veor_u32(sign, x);
		XMVECTOR localQ1 = veor_u32(sign, q1);
#endif
		XMVECTOR xm1 = XMVectorSubtract(x, one);
		XMVECTOR splatD = XMVectorSubtract(one, splatT);
		XMVECTOR cT = XMQuaternionSlerpCoefficient(splatT, xm1);
		XMVECTOR cD = XMQuaternionSlerpCoefficient(splatD, xm1);
		cT = XMVectorMultiply(cT, localQ1);
		cD = _DXMEXT XMVectorMultiplyAdd(cD, q0, cT);
		return cD;
#endif
	}

	// Caculate the rotation quaternion base on v1 and v2 (shortest rotation geo-distance in sphere surface)
	inline XMVECTOR XM_CALLCONV XMQuaternionRotationVectorToVector(FXMVECTOR v1, FXMVECTOR v2) {
		assert(!XMVector3Equal(v1, XMVectorZero()));
		assert(!XMVector3Equal(v2, XMVectorZero()));
		XMVECTOR n1 = XMVector3Normalize(v1);
		XMVECTOR n2 = XMVector3Normalize(v2);
		XMVECTOR epsilon = g_XMEpsilon.v;

		if (XMVector4NearEqual(n1, n2, epsilon))
		{
			return XMQuaternionIdentity();
		}

		XMVECTOR axias = XMVector3Cross(n1, n2);
		if (XMVector4NearEqual(axias, XMVectorZero(), epsilon))
		{
			n2 = g_XMIdentityR0.v;
			axias = XMVector3Cross(n1, n2);
			if (XMVector4NearEqual(axias, XMVectorZero(), epsilon))
			{
				n2 = g_XMIdentityR1.v;
				axias = XMVector3Cross(n1, n2);
			}
		}
		float angle = XMScalarACos(XMVectorGetX(XMVector3Dot(n1, n2)));
		auto rot = XMQuaternionRotationAxis(axias, angle);
		return rot;
	}

	// Caculate Left handed eular angles with 
	// rotation sequence R(Z)-R(Y)-R(X), where patch is Y
	// https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
	inline XMVECTOR XM_CALLCONV XMQuaternionEulerAngleZYX(FXMVECTOR q)
	{
#if defined(_XM_NO_INTRINSICS_)
		float q0 = q.vector4_f32[3], q1 = q.vector4_f32[0], q2 = q.vector4_f32[1], q3 = q.vector4_f32[2];
		XMVECTOR euler;
		euler.vector4_f32[0] = atan2(2 * (q0*q1 + q2*q3), 1 - 2 * (q1*q1 + q2*q2));
		euler.vector4_f32[1] = asinf(2 * (q0*q2 - q3*q1));
		euler.vector4_f32[2] = atan2(2 * (q0*q3 + q2*q1), 1 - 2 * (q2*q2 + q3*q3));
		euler.vector4_f32[3] = 0;
		return euler;
#elif defined(_XM_SSE_INTRINSICS_) || defined(_XM_ARM_NEON_INTRINSICS_)

		//float q0 = q.m128_f32[3], q1 = q.m128_f32[0], q2 = q.m128_f32[1], q3 = q.m128_f32[2];
		//XMVECTOR euler;
		//euler.m128_f32[0] = atan2(2 * (q0*q1 + q2*q3), 1 - 2 * (q1*q1 + q2*q2));
		//euler.m128_f32[1] = asinf(2 * (q0*q2 - q3*q1));
		//euler.m128_f32[2] = atan2(2 * (q0*q3 + q2*q1), 1 - 2 * (q2*q2 + q3*q3));
		//euler.m128_f32[3] = 0;
		//return euler;

		XMVECTOR q02 = _DXMEXT XMVectorSwizzle<3, 3, 1, 1>(q); // q0022
		XMVECTOR q13 = _DXMEXT XMVectorSwizzle<0, 0, 2, 2>(q); // q1133

													   // eular X - Roll
		XMVECTOR Y = _DXMEXT XMVector4Dot(q02, q13);	// 2 * (q2*q1+q0*q3)
		XMVECTOR X = _DXMEXT XMVectorSwizzle<0, 0, 1, 1>(q); // q2121
		X = _DXMEXT XMVector4Dot(X, X);					// 2 * (q2*q2+q1*q1)
		X = XMVectorSubtract(g_XMOne.v, X);		// 1 - 2 * (q2*q2+q1*q1)
		XMVECTOR vResult = XMVectorATan2(Y, X);

		// eular Y - Y
		q13 = _DXMEXT XMVectorSwizzle<2, 2, 0, 0>(q); // now q3311
		Y = _DXMEXT XMVector4Dot(q02, q13);			  // now 3311 dot 0022 = q3*q0 + q1*q2
		X = _DXMEXT XMVectorSwizzle<1, 1, 2, 2>(q);		  // now q3232
		X = _DXMEXT XMVector4Dot(X, X);				  // now 2 * (q3*q3+q2*q2)
		X = XMVectorSubtract(g_XMOne.v, X);	  // now 1 - 2 * (q3*q3+q2*q2)
		X = XMVectorATan2(Y, X);

		// eular X - Patch
		Y = _DXMEXT XMVectorSwizzle<2, 3, 0, 1>(q); // now q3 q0 q1 q2 
		Y = XMVectorMultiply(Y, g_XMNegtiveXZ.v); // now -q3 q0 -q1 q2
		Y = _DXMEXT XMVector4Dot(Y, q); // now -q1*q3 + q2*q0 - q3*q1 + q0*q2
		Y = XMVectorASin(Y);

		// merge result
		X = XMVectorMergeXY(X, Y);
		vResult = XMVectorSelect(X, vResult, g_XMSelect1000.v);
		vResult = XMVectorAndInt(vResult, g_XMMask3.v);
		return vResult;
#endif
	}

	XMVECTOR XM_CALLCONV XMVector3Displacement(FXMVECTOR V, FXMVECTOR RotationQuaternion, FXMVECTOR TranslationQuaternion);

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionTranslation(FXMVECTOR T)
	{
		static const XMVECTORF32 Control = { 0.5f,0.5f,0.5f,.0f };
		XMVECTOR Qe = XMVectorMultiply(T, Control);
		XMVECTOR Qr = XMQuaternionIdentity();
		XMDUALVECTOR dqRes;
		dqRes.r[0] = Qr;
		dqRes.r[1] = Qe;
		return dqRes;
	}

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionRotation(FXMVECTOR Q)
	{
		XMDUALVECTOR dqRes;
		dqRes.r[0] = Q;
		dqRes.r[1] = XMVectorZero();
		return dqRes;
	}

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionRigidTransform(FXMVECTOR RotationOrigin, FXMVECTOR RotationQuaternion, FXMVECTOR Translation)
	{
		static const XMVECTORF32 Control = { 0.5f,0.5f,0.5f,.0f };

		XMVECTOR Qr = RotationQuaternion;
		XMVECTOR Qe = RotationOrigin + Translation;
		Qe = XMVectorMultiply(Qe, Control);
		XMVECTOR Qo = XMVectorMultiply(RotationOrigin, Control);
		Qo = XMVectorNegate(Qo);

		Qe = XMQuaternionMultiply(Qr, Qe);
		Qo = XMQuaternionMultiply(Qo, Qr);
		Qe += Qo;

		XMDUALVECTOR dqRes;
		dqRes.r[0] = Qr;
		dqRes.r[1] = Qe;
		return dqRes;
	}

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionRotationTranslation(FXMVECTOR Q, FXMVECTOR T)
	{
		// non-dual part (just copy q0):
		// dual part:
		static const XMVECTORF32 Control = { 0.5f,0.5f,0.5f,.0f };
		XMVECTOR Qe = XMVectorMultiply(T, Control);
		Qe = XMQuaternionMultiply(Q, Qe);
		XMDUALVECTOR dqRes;
		dqRes.r[0] = Q;
		dqRes.r[1] = Qe;

		return dqRes;
	};

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionNormalize(FXMDUALVECTOR Dq)
	{
		XMVECTOR Length = _DXMEXT XMVector4Length(Dq.r[0]);
		XMDUALVECTOR dqRes;
		dqRes.r[0] = XMVectorDivide(Dq.r[0], Length);
		dqRes.r[1] = XMVectorDivide(Dq.r[1], Length);
		return dqRes;
	}
	inline XMVECTOR XM_CALLCONV XMDualQuaternionNormalizeEst(FXMDUALVECTOR Dq)
	{
		return _DXMEXT XMVector4NormalizeEst(Dq.r[0]);
	}
	inline XMVECTOR XM_CALLCONV XMDualQuaternionLength(FXMDUALVECTOR Dq)
	{
		return _DXMEXT XMVector4Length(Dq.r[0]);
	}
	inline XMVECTOR XM_CALLCONV XMDualQuaternionLengthSq(FXMDUALVECTOR Dq)
	{
		return _DXMEXT XMVector4LengthSq(Dq.r[0]);
	}
	inline XMVECTOR XM_CALLCONV XMDualQuaternionLengthEst(FXMDUALVECTOR Dq)
	{
		return _DXMEXT XMVector4LengthEst(Dq.r[0]);
	}
	inline XMDUALVECTOR XM_CALLCONV XMDualVectorConjugate(FXMDUALVECTOR Dq)
	{
		XMDUALVECTOR dvResult(Dq.r[0], XMVectorNegate(Dq.r[1]));
		return dvResult;
	}
	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionConjugate(FXMDUALVECTOR Dq)
	{
		XMDUALVECTOR dvResult;
		dvResult.r[0] = XMQuaternionConjugate(Dq.r[0]);
		dvResult.r[1] = XMQuaternionConjugate(Dq.r[1]);
		return dvResult;
	}

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionInverse(FXMDUALVECTOR Dq)
	{
		XMDUALVECTOR dvResult = XMDualQuaternionConjugate(Dq);
		dvResult.r[0] = XMQuaternionNormalize(dvResult.r[0]);
	}

	inline XMDUALVECTOR XM_CALLCONV XMDualQuaternionMultipy(FXMDUALVECTOR DQ0, GXMDUALVECTOR DQ1)
	{
		XMDUALVECTOR dvResult;
		dvResult.r[0] = XMQuaternionMultiply(DQ0.r[0], DQ1.r[0]);
		dvResult.r[1] = XMQuaternionMultiply(DQ0.r[0], DQ1.r[1]);
		XMVECTOR temp = XMQuaternionMultiply(DQ0.r[1], DQ1.r[0]);
		dvResult.r[1] += temp;
		return dvResult;
	}

	inline bool XM_CALLCONV XMDualQuaternionDecompose(XMVECTOR* outRotQuat, XMVECTOR* outTrans, FXMDUALVECTOR Dq)
	{
		static const XMVECTORF32 ControlX = { 2.0f,-2.0f,2.0f,-2.0f };
		static const XMVECTORF32 ControlY = { 2.0f,2.0f,-2.0f,-2.0f };
		static const XMVECTORF32 ControlZ = { -2.0f,2.0f,2.0f,-2.0f };

		//vT.x = 2.0f*( Qe.x*Q.w - Qe.y*Q.z + Qe.z*Q.y - Qe.w*Q.x);
		//vT.y = 2.0f*( Qe.x*Q.z + Qe.y*Q.w - Qe.z*Q.x - Qe.w*Q.y);
		//vT.z = 2.0f*(-Qe.x*Q.y + Qe.y*Q.x + Qe.z*Q.w - Qe.w*Q.z);
		XMVECTOR vT;
		XMVECTOR Qr = Dq.r[0];
		XMVECTOR Qe = Dq.r[1];
		Qr = _DXMEXT XMVector4Normalize(Qr);

		XMVECTOR Q = _DXMEXT XMVectorSwizzle<3, 2, 1, 0>(Qr);
		Q = XMVectorMultiply(Qe, Q);
		Q = _DXMEXT XMVector4Dot(Q, ControlX.v);
		vT = XMVectorAndInt(Q, g_XMMaskX.v);

		Q = _DXMEXT XMVectorSwizzle<2, 3, 0, 1>(Qr);
		Q = XMVectorMultiply(Qe, Q);
		Q = _DXMEXT XMVector4Dot(Q, ControlY.v);
		Q = XMVectorAndInt(Q, g_XMMaskY.v);
		vT = XMVectorOrInt(vT, Q);

		Q = _DXMEXT XMVectorSwizzle<1, 0, 3, 2>(Qr);
		Q = XMVectorMultiply(Qe, Q);
		Q = _DXMEXT XMVector4Dot(Q, ControlZ.v);
		Q = XMVectorAndInt(Q, g_XMMaskZ.v);
		vT = XMVectorOrInt(vT, Q);

		*outRotQuat = Qr;
		*outTrans = vT;
		return true;
	}

	inline XMVECTOR XM_CALLCONV XMVector3Displacement(FXMVECTOR V, FXMVECTOR RotationQuaternion, FXMVECTOR TranslationQuaternion)
	{
		XMVECTOR Qr = RotationQuaternion;
		XMVECTOR Qe = TranslationQuaternion;
		assert("The dual quaternion is invaliad: the dual part must be zero," && XMVector4NearEqual(XMVector4Dot(Qr, Qe), XMVectorZero(), XMVectorReplicate(0.0001f)));

		XMVECTOR vRes = XMVector3Rotate(V, Qr);
		Qr = XMQuaternionConjugate(Qr);
		XMVECTOR vTrans = XMQuaternionMultiply(Qr, Qe);
		//XMVECTOR vTrans = XMVectorSplatW(Qr) * Qe - Qr * XMVectorSplatW(Qe) + XMVector3Cross(Qr,Qe);
		vTrans *= g_XMTwo.v;

		vRes += vTrans;
		return vRes;
	}


	inline XMVECTOR CaculatePrincipleAxisFromPoints(size_t Count, const XMFLOAT3 * pPoints, size_t Stride, float pointsUpperBound)
	{
		XMVECTOR InvScale = XMVectorReplicate(1.0f / pointsUpperBound);
		XMVECTOR CenterOfMass = XMVectorZero();

		// Compute the center of mass and inertia tensor of the points.
		for (size_t i = 0; i < Count; ++i)
		{
			XMVECTOR Point = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(reinterpret_cast<const uint8_t*>(pPoints) + i * Stride));

			CenterOfMass += Point * InvScale; // Only Modify is here, normalize scales to 1 to prevent float overflow
		}

		CenterOfMass *= XMVectorReciprocal(XMVectorReplicate(float(Count)));
		XMVECTOR NegCenterOfMass = -CenterOfMass;

		// Compute the inertia tensor of the points around the center of mass.
		// Using the center of mass is not strictly necessary, but will hopefully
		// improve the stability of finding the eigenvectors.
		XMVECTOR XX_YY_ZZ = XMVectorZero();
		XMVECTOR XY_XZ_YZ = XMVectorZero();

		for (size_t i = 0; i < Count; ++i)
		{
			XMVECTOR Point = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(reinterpret_cast<const uint8_t*>(pPoints) + i * Stride));
			Point = XMVectorMultiplyAdd(Point, InvScale, NegCenterOfMass); // normalize scales to 1 to prevent float overflow

			XX_YY_ZZ += Point * Point;

			XMVECTOR XXY = XMVectorSwizzle<XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_W>(Point);
			XMVECTOR YZZ = XMVectorSwizzle<XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_W>(Point);

			XY_XZ_YZ += XXY * YZZ;
		}

		XMVECTOR v1, v2, v3;

		// Compute the eigenvectors of the inertia tensor.
		DirectX::Internal::CalculateEigenVectorsFromCovarianceMatrix(XMVectorGetX(XX_YY_ZZ), XMVectorGetY(XX_YY_ZZ),
			XMVectorGetZ(XX_YY_ZZ),
			XMVectorGetX(XY_XZ_YZ), XMVectorGetY(XY_XZ_YZ),
			XMVectorGetZ(XY_XZ_YZ),
			&v1, &v2, &v3);

		// Put them in a matrix.
		XMMATRIX R;

		R.r[0] = XMVectorSetW(v1, 0.f);
		R.r[1] = XMVectorSetW(v2, 0.f);
		R.r[2] = XMVectorSetW(v3, 0.f);
		R.r[3] = g_XMIdentityR3.v;

		// Multiply by -1 to convert the matrix into a right handed coordinate 
		// system (Det ~= 1) in case the eigenvectors form a left handed 
		// coordinate system (Det ~= -1) because XMQuaternionRotationMatrix only 
		// works on right handed matrices.
		XMVECTOR Det = XMMatrixDeterminant(R);

		if (XMVector4Less(Det, XMVectorZero()))
		{
			R.r[0] *= g_XMNegativeOne.v;
			R.r[1] *= g_XMNegativeOne.v;
			R.r[2] *= g_XMNegativeOne.v;
		}

		// Get the rotation quaternion from the matrix.
		XMVECTOR vOrientation = XMQuaternionRotationMatrix(R);

		// Make sure it is normal (in case the vectors are slightly non-orthogonal).
		vOrientation = XMQuaternionNormalize(vOrientation);

		return vOrientation;
		//// Rebuild the rotation matrix from the quaternion.
		//R = XMMatrixRotationQuaternion(vOrientation);

		//// Build the rotation into the rotated space.
		//XMMATRIX InverseR = XMMatrixTranspose(R);
	}

	// Create AABB & OBB from points, also normalize points extents to prevent float overflow during caculating principle axis
	inline void CreateBoundingBoxesFromPoints(BoundingBox & aabb, BoundingOrientedBox & obb, size_t Count, const XMFLOAT3 * pPoints, size_t Stride) {
		BoundingBox::CreateFromPoints(aabb, Count, pPoints, Stride);
		float pointsUpperBound = XMMax(XMMax(aabb.Extents.x, aabb.Extents.y), aabb.Extents.z);


		XMVECTOR vOrientation = CaculatePrincipleAxisFromPoints(Count, pPoints, Stride, pointsUpperBound);

		// Rebuild the rotation matrix from the quaternion.
		XMMATRIX R = XMMatrixRotationQuaternion(vOrientation);

		// Build the rotation into the rotated space.
		XMMATRIX InverseR = XMMatrixTranspose(R);

		// Find the minimum OBB using the eigenvectors as the axes.
		XMVECTOR vMin, vMax;

		vMin = vMax = XMVector3TransformNormal(XMLoadFloat3(pPoints), InverseR);

		for (size_t i = 1; i < Count; ++i)
		{
			XMVECTOR Point = XMVector3TransformNormal(XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(reinterpret_cast<const uint8_t*>(pPoints) + i * Stride)),
				InverseR);

			vMin = XMVectorMin(vMin, Point);
			vMax = XMVectorMax(vMax, Point);
		}

		// Rotate the center into world space.
		XMVECTOR vCenter = (vMin + vMax) * 0.5f;
		vCenter = XMVector3TransformNormal(vCenter, R);

		// Store center, extents, and orientation.
		XMStoreFloat3(&obb.Center, vCenter);
		XMStoreFloat3(&obb.Extents, (vMax - vMin) * 0.5f);
		XMStoreFloat4(&obb.Orientation, vOrientation);
	}

	inline XMMATRIX XM_CALLCONV XMMatrixScalingFromCenter(FXMVECTOR scale, FXMVECTOR center)
	{
		XMMATRIX m = XMMatrixScalingFromVector(scale);
		m.r[3] = XMVectorSubtract(g_XMOne.v, scale);
		m.r[3] *= center;
		m.r[3] = XMVectorSelect(g_XMOne.v, m.r[3], g_XMSelect1110.v);
		return m;
	}

	// No Matrix Multiply inside, significant faster than affine transform
	inline XMMATRIX XM_CALLCONV XMMatrixRigidTransform(FXMVECTOR RotationOrigin, FXMVECTOR RotationQuaterion, FXMVECTOR Translation)
	{
		XMVECTOR VTranslation = XMVector3Rotate(RotationOrigin, RotationQuaterion);
		VTranslation = XMVectorAdd(VTranslation, RotationOrigin);
		VTranslation = XMVectorAdd(VTranslation, VTranslation);
		VTranslation = XMVectorSelect(g_XMSelect1110.v, Translation, g_XMSelect1110.v);
		XMMATRIX M = XMMatrixRotationQuaternion(RotationQuaterion);
		M.r[3] = XMVectorAdd(M.r[3], VTranslation);
		return M;
	}

	// No Matrix Multiply inside, significant faster than affine transform
	inline XMMATRIX XM_CALLCONV XMMatrixRigidTransform(FXMVECTOR RotationQuaterion, FXMVECTOR Translation)
	{
		XMMATRIX M = XMMatrixRotationQuaternion(RotationQuaterion);
		XMVECTOR VTranslation = XMVectorSelect(g_XMSelect1110.v, Translation, g_XMSelect1110.v);
		M.r[3] = XMVectorAdd(M.r[3], VTranslation);
		return M;
	}

	inline XMVECTOR XM_CALLCONV XMVector3Displacement(FXMVECTOR V, CXMDUALVECTOR TransformDualQuaternion)
	{
		return XMVector3Displacement(V, TransformDualQuaternion.r[0], TransformDualQuaternion.r[1]);
	}

#ifdef _MSC_VER
#pragma endregion
#endif

#ifdef _MSC_VER
#pragma region Quaternion Eular Angle Convertions
#endif

	namespace Internal
	{
		// 3 Permuation contains 2-Permutation (XY)(YZ)(ZX), which are self-inverse
		// Identity (XYZ) are also self-inverse,Special case (YZX) and (ZXY) list below
		template <unsigned _X, unsigned _Y, unsigned _Z>
		struct SwizzleInverse
		{
			static const unsigned X = _X;
			static const unsigned Y = _Y;
			static const unsigned Z = _Z;
		};
		template <>
		struct SwizzleInverse<1, 2, 0>
		{
			static const unsigned X = 2;
			static const unsigned Y = 0;
			static const unsigned Z = 1;
		};
		template <>
		struct SwizzleInverse<2, 0, 1>
		{
			static const unsigned X = 1;
			static const unsigned Y = 2;
			static const unsigned Z = 0;
		};
	}

	enum AxisEnum
	{
		AxisX = 0,
		AxisY = 1,
		AxisZ = 2,
		AxisPatch = AxisX,
		AxisYaw = AxisY,
		AxisRoll = AxisZ
	};

	template <unsigned X, unsigned Y, unsigned Z>
	// Caculate Left handed eular angles with 
	// rotation sequence R(X)-R(Y)-R(Z)
	XMVECTOR XM_CALLCONV XMQuaternionEulerAngleABC(FXMVECTOR q);


	template <unsigned X, unsigned Y, unsigned Z >
	// Caculate Left handed eular angles with 3 different rotation axis
	// rotation sequence R(X)-R(Y)-R(Z)	
	inline XMVECTOR XM_CALLCONV XMQuaternionEulerAngleABC(FXMVECTOR q)
	{
		static_assert(X <= 2 && Y <= 2 && Z <= 2 && X != Y && Z != Y && X != Z, "X-Y-Z must be different Axis");

		XMVECTOR qs = XMVectorSwizzle<X, Y, Z, 3>(q);
		qs = XMQuaternionEulerAngleZYX(qs);
		using InvS = Internal::SwizzleInverse<X, Y, Z>;
		qs = XMVectorSwizzle<InvS::X, InvS::Y, InvS::Z, 3>(qs);
		return qs;
	}

	template <unsigned X, unsigned Y>
	// Caculate Left handed eular angles with 2 different rotation axis
	// rotation sequence R(X)-R(Y)-R(X)	
	inline XMVECTOR XM_CALLCONV XMQuaternionEulerAngleABA(FXMVECTOR q)
	{
	}

	// Left handed eular angles
	// rotation sequence R(Y)-R(X)-R(Z), so called Yaw-Pitch-Roll
	// return : <R(X),R(Y),R(Z),0>
	inline XMVECTOR XM_CALLCONV XMQuaternionEulerAngleYXZ(FXMVECTOR q)
	{
		return XMQuaternionEulerAngleABC<AxisY, AxisX, AxisZ>(q);
	}

	// left handed eular angles : <Pitch,Yaw,Roll,0>, ready with XMQuaternionRotationYawPitchRoll
	// rotation sequence R(Y)-R(X)-R(Z), so called Yaw-Pitch-Roll
	inline XMVECTOR XM_CALLCONV XMQuaternionEulerAngleYawPitchRoll(FXMVECTOR q)
	{
		return XMQuaternionEulerAngleABC<AxisY, AxisX, AxisZ>(q);
	}

	// Rotate along Y-axis
	inline XMVECTOR XMQuaternionRotationYaw(float yaw)
	{
		return XMQuaternionRotationNormal(g_XMIdentityR1.v, yaw);
	}

	// Rotate along X-axis
	inline XMVECTOR XMQuaternionRotationPatch(float patch)
	{
		return XMQuaternionRotationNormal(g_XMIdentityR0.v, patch);
	}

	// Rotate along Z-axis
	inline XMVECTOR XMQuaternionRotationRoll(float roll)
	{
		return XMQuaternionRotationNormal(g_XMIdentityR2.v, roll);
	}

#ifdef _MSC_VER
#pragma endregion
#endif

	namespace Internal
	{
		template <typename T>
		inline void Swap(T& x, T&y)
		{
			T t = x;
			x = y;
			y = t;
		}
	}
	// This function garuntee the extends of the bounding box is sorted from bigger to smaller
	inline void CreateBoundingOrientedBoxFromPoints(_Out_ BoundingOrientedBox& Out, _In_ size_t Count,
		_In_reads_bytes_(sizeof(XMFLOAT3) + Stride*(Count - 1)) const XMFLOAT3* pPoints, _In_ size_t Stride) {
		BoundingOrientedBox::CreateFromPoints(Out, Count, pPoints, Stride);
		XMMATRIX rot = XMMatrixIdentity();
		auto& ext = Out.Extents;
		using Internal::Swap;

		// Sort the demension
		if (ext.x < ext.y)
		{
			Swap(ext.x, ext.y);
			Swap(rot.r[0], rot.r[1]);
			rot.r[2] = -rot.r[2]; // Flip the other axias to maintain the chirality
		}
		if (ext.y < ext.z)
		{
			Swap(ext.y, ext.z);
			Swap(rot.r[1], rot.r[2]);
			rot.r[0] = -rot.r[0];
		}
		if (ext.x < ext.y)
		{
			Swap(ext.x, ext.y);
			Swap(rot.r[0], rot.r[1]);
			rot.r[2] = -rot.r[2];
		}

		//qw = sqrt(1 + m00 + m11 + m22) / 2
		//qx = (m21 - m12) / (4 * qw)
		//qy = (m02 - m20) / (4 * qw)
		//qz = (m10 - m01) / (4 * qw)
		XMVECTOR q = XMQuaternionRotationMatrix(rot);
		XMStoreFloat4(&Out.Orientation, XMQuaternionMultiply(q, XMLoadFloat4(&Out.Orientation)));
		//auto q = XMQuaternionRotationMatrix(rot);
	}

	inline XMVECTOR CaculatePrincipleAxisFromPoints(_In_ size_t Count,
		_In_reads_bytes_(sizeof(XMFLOAT3) + Stride*(Count - 1)) const XMFLOAT3* pPoints, _In_ size_t Stride, float pointsUpperBound);

	// Create AABB & OBB from points, also normalize points extents to prevent float overflow during caculating principle axis
	inline void CreateBoundingBoxesFromPoints(_Out_ BoundingBox& aabb, _Out_ BoundingOrientedBox& obb, _In_ size_t Count,
		_In_reads_bytes_(sizeof(XMFLOAT3) + Stride*(Count - 1)) const XMFLOAT3* pPoints, _In_ size_t Stride);

	inline XMMATRIX XM_CALLCONV XMMatrixScalingFromCenter(FXMVECTOR scale, FXMVECTOR center);

	// No Matrix Multiply inside, significant faster than affine transform
	inline XMMATRIX XM_CALLCONV XMMatrixRigidTransform(FXMVECTOR RotationOrigin, FXMVECTOR RotationQuaterion, FXMVECTOR Translation);

	// No Matrix Multiply inside, significant faster than affine transform
	inline XMMATRIX XM_CALLCONV XMMatrixRigidTransform(FXMVECTOR RotationQuaterion, FXMVECTOR Translation);

#pragma region XMLoadStoreA Helpers
	inline XMVECTOR XMLoad(const XMFLOAT2& src)
	{
		return XMLoadFloat2(&src);
	}
	inline XMVECTOR XMLoadA(const XMFLOAT2& src)
	{
		return XMLoadFloat2A(reinterpret_cast<const XMFLOAT2A*>(&src));
	}
	inline void XM_CALLCONV XMStore(XMFLOAT2& dest, FXMVECTOR v)
	{
		XMStoreFloat2(&dest, v);
	}
	inline void XM_CALLCONV XMStoreA(XMFLOAT2& dest, FXMVECTOR v)
	{
		XMStoreFloat2A(reinterpret_cast<XMFLOAT2A*>(&dest), v);
	}
	inline XMVECTOR XMLoad(const XMFLOAT3& src)
	{
		return XMLoadFloat3(&src);
	}
	inline XMVECTOR XMLoadA(const XMFLOAT3& src)
	{
		return XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(&src));
	}
	inline void XM_CALLCONV XMStore(XMFLOAT3& dest, FXMVECTOR v)
	{
		XMStoreFloat3(&dest, v);
	}
	inline void XM_CALLCONV XMStoreA(XMFLOAT3& dest, FXMVECTOR v)
	{
		XMStoreFloat3A(reinterpret_cast<XMFLOAT3A*>(&dest), v);
	}
	inline XMVECTOR XMLoad(const XMFLOAT4& src)
	{
		return XMLoadFloat4(&src);
	}
	inline XMVECTOR XMLoadA(const XMFLOAT4& src)
	{
		return XMLoadFloat4A(reinterpret_cast<const XMFLOAT4A*>(&src));
	}
	inline void XM_CALLCONV XMStore(XMFLOAT4& dest, FXMVECTOR v)
	{
		XMStoreFloat4(&dest, v);
	}
	inline void XM_CALLCONV XMStoreA(XMFLOAT4& dest, FXMVECTOR v)
	{
		XMStoreFloat4A(reinterpret_cast<XMFLOAT4A*>(&dest), v);
	}

	inline XMMATRIX XMLoad(const XMFLOAT4X4& src)
	{
		return XMLoadFloat4x4(&src);
	}
	inline XMMATRIX XMLoadA(const XMFLOAT4X4& src)
	{
		return XMLoadFloat4x4A(reinterpret_cast<const XMFLOAT4X4A*>(&src));
	}
	inline void XM_CALLCONV XMStore(XMFLOAT4X4& dest, FXMMATRIX m)
	{
		XMStoreFloat4x4(&dest, m);
	}
	inline void XM_CALLCONV XMStoreA(XMFLOAT4X4& dest, FXMMATRIX m)
	{
		XMStoreFloat4x4A(reinterpret_cast<XMFLOAT4X4A*>(&dest), m);
	}
#pragma endregion

	enum BoundingGeometryType
	{
		Geometry_Unknown,
		Geometry_AxisAlignedBox,
		Geometry_OrientedBox,
		Geometry_Frustum,
		Geometry_Sphere,
	};

	struct BoundingGeometry
	{
		BoundingGeometryType Type;
		union
		{
			BoundingBox				AxisAlignedBox;
			BoundingOrientedBox		OrientedBox;
			BoundingFrustum			Frustum;
			BoundingSphere			Sphere;
		};

		BoundingGeometry();

		explicit BoundingGeometry(const BoundingBox& aabb);
		explicit BoundingGeometry(const BoundingOrientedBox& obox);
		explicit BoundingGeometry(const BoundingFrustum& frustum);
		explicit BoundingGeometry(const BoundingSphere& sphere);

		BoundingGeometry& operator=(const BoundingGeometry& rhs) = default;
		BoundingGeometry& operator=(BoundingGeometry&& rhs) = default;
		BoundingGeometry& operator=(const BoundingBox& rhs);
		BoundingGeometry& operator=(const BoundingOrientedBox& rhs);
		BoundingGeometry& operator=(const BoundingFrustum& rhs);
		BoundingGeometry& operator=(const BoundingSphere& rhs);

		ContainmentType XM_CALLCONV Contains(_In_ FXMVECTOR Point) const;
		ContainmentType XM_CALLCONV Contains(_In_ FXMVECTOR V0, _In_ FXMVECTOR V1, _In_ FXMVECTOR V2) const;

		template <typename GeometryType>
		ContainmentType ContainsGeometry(_In_ const GeometryType& geometry) const;

		ContainmentType Contains(_In_ const BoundingSphere& geometry) const;
		ContainmentType Contains(_In_ const BoundingBox& geometry) const;
		ContainmentType Contains(_In_ const BoundingOrientedBox& geometry) const;
		ContainmentType Contains(_In_ const BoundingFrustum& geometry) const;
		ContainmentType Contains(_In_ const BoundingGeometry& geometry) const;

		// Six-Plane containment test
		ContainmentType XM_CALLCONV ContainedBy(_In_ FXMVECTOR Plane0, _In_ FXMVECTOR Plane1, _In_ FXMVECTOR Plane2,
			_In_ GXMVECTOR Plane3, _In_ HXMVECTOR Plane4, _In_ HXMVECTOR Plane5) const;

		template <typename GeometryType>
		bool IntersectsGeometry(_In_ const GeometryType& geometry) const;

		bool Intersects(_In_ const BoundingSphere& geometry) const;
		bool Intersects(_In_ const BoundingBox& geometry) const;
		bool Intersects(_In_ const BoundingOrientedBox& geometry) const;
		bool Intersects(_In_ const BoundingFrustum& geometry) const;
		bool Intersects(_In_ const BoundingGeometry& geometry) const;

		// Triangle test
		bool XM_CALLCONV Intersects(_In_ FXMVECTOR V0, _In_ FXMVECTOR V1, _In_ FXMVECTOR V2) const;
		// Ray test
		bool XM_CALLCONV Intersects(_In_ FXMVECTOR Origin, _In_ FXMVECTOR Direction, _Out_ float& Dist) const;
		// Plane test
		PlaneIntersectionType XM_CALLCONV	Intersects(_In_ FXMVECTOR Plane) const;

		void XM_CALLCONV Transform(_Out_ BoundingGeometry& Out, _In_ FXMMATRIX M) const;
		void XM_CALLCONV Transform(_Out_ BoundingGeometry& Out, _In_ float Scale, _In_ FXMVECTOR Rotation, _In_ FXMVECTOR Translation) const;
	};


	namespace BoundingFrustumExtension
	{
		inline void CreateFromMatrixRH(BoundingFrustum& Out, CXMMATRIX Projection);
	}

	//Some supporting method

	/// <summary>
	/// 
	/// </summary>
	/// <param name="V0"></param>
	/// <param name="V1"></param>
	/// <param name="V2"></param>
	/// <param name="BC"></param>
	/// <returns></returns>
	inline XMVECTOR XM_CALLCONV XMVectorBaryCentricV(FXMVECTOR V0, FXMVECTOR V1, FXMVECTOR V2, GXMVECTOR BC)
	{
		XMVECTOR V;
		XMVECTOR T = XMVectorSplatX(BC);
		V = XMVectorMultiply(V0, T);
		T = XMVectorSplatY(BC);
		V = XMVectorMultiplyAdd(V1, T, V);
		T = XMVectorSplatZ(BC);
		V = XMVectorMultiplyAdd(V2, T, V);
		return V;
	}

	namespace TriangleTests
	{
		/// <summary>
		/// return the barycentric coordinate of position P inside triangle V0,V1,V2 ,
		/// so that V0 * bc.x + V1 * bc.y + V2 * bc.z = P ,
		/// bc.w are unspecified
		/// </summary>
		/// <param name="P">Position inside triangle</param>
		/// <param name="V0"></param>
		/// <param name="V1"></param>
		/// <param name="V2"></param>
		/// <returns></returns>
		inline XMVECTOR XM_CALLCONV BarycentricCoordinate(FXMVECTOR P, FXMVECTOR V0, FXMVECTOR V1, GXMVECTOR V2)
		{
			XMVECTOR E = V2 - V1;
			XMVECTOR R0 = P - V1;
			R0 = XMVector3Cross(R0, E);
			R0 = XMVector3Length(R0);

			E = V0 - V2;
			XMVECTOR R1 = P - V2;
			R1 = XMVector3Cross(R1, E);
			R1 = XMVector3Length(R1);

			E = V1 - V0;
			XMVECTOR R2 = P - V0;
			R2 = XMVector3Cross(R2, E);
			R2 = XMVector3Length(R2);

			E = R0 + R1 + R2;
			R0 = XMVectorSelect(R1, R0, g_XMSelect1000.v); // r0 r1 r1 r1
			R0 = XMVectorSelect(R2, R0, g_XMSelect1100.v); // r1 r1 r2 r2
			R0 /= E;

			XMVECTOR PR = XMVectorBaryCentricV(V0, V1, V2, R0);
			PR -= P;
			assert(XMVectorGetX(XMVector3Length(PR)) < 0.001f);
			return R0;
		}
	}

	namespace LineSegmentTest
	{
		inline bool Intersects2D(FXMVECTOR A0, FXMVECTOR A1, FXMVECTOR B0, GXMVECTOR B1, XMVECTOR* pIntersection = nullptr);

		inline float XM_CALLCONV Distance(FXMVECTOR p, FXMVECTOR s0, FXMVECTOR s1);
		inline float XM_CALLCONV Distance(FXMVECTOR p, const XMFLOAT3 *path, size_t nPoint, size_t strideInByte = sizeof(XMFLOAT3));

		// Takes a space point and space line segment , return the projection point on the line segment
		//  A0  |		A1		  |
		//      |s0-------------s1|
		//      |				  |		A2
		// where p in area of A0 returns s0 , area A2 returns s1 , point in A1 returns the really projection point 
		inline XMVECTOR XM_CALLCONV Projection(FXMVECTOR p, FXMVECTOR s0, FXMVECTOR s1);

		template <typename T>
		inline XMVECTOR XM_CALLCONV Projection(FXMVECTOR p, const T *path, size_t nPoint, size_t strideInByte = sizeof(T));
	}

}

#include "DirectXMathExtend.inl"

// Extending std lib for output
#ifdef _OSTREAM_
#include <iomanip>
namespace DirectX
{
	inline std::ostream& operator << (std::ostream& lhs, const DirectX::XMFLOAT2& rhs)
	{
		lhs << '(' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.x
			<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.y << ')';
		return lhs;
	};

	inline std::ostream& operator << (std::ostream& lhs, const DirectX::XMFLOAT3& rhs)
	{
		lhs << '(' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.x
			<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.y
			<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.z << ')';
		return lhs;
	};

	inline std::ostream& operator << (std::ostream& lhs, const DirectX::XMFLOAT4& rhs)
	{
		lhs << '(' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.x
			<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.y
			<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.z
			<< ',' << std::setw(6) << setiosflags(std::ios::fixed) << std::setprecision(3) << rhs.w << ')';
		return lhs;
	};
}

#endif

#include "DirectXMathSimpleVectors.h"
#include "DirectXMathTransforms.h"
#endif