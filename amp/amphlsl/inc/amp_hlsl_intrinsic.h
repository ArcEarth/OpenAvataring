#pragma once
#include <amp_short_vectors.h>
#include <amp_math.h>
// customized header for optimzed fmax/fmin
#include <minmax>
#include "amp_short_matrix.h"
#define _AMP_HLSL_INTRINSIC_H

#define __Out_Param__(T,name) T& name
#define __Inout_Param__(T,name) T& name

namespace Concurrency
{
	constexpr float XM_PI = 3.141592654f;
	constexpr float XM_2PI = 6.283185307f;
	constexpr float XM_1DIVPI = 0.318309886f;
	constexpr float XM_1DIV2PI = 0.159154943f;
	constexpr float XM_PIDIV2 = 1.570796327f;
	constexpr float XM_PIDIV4 = 0.785398163f;

	namespace direct3d
	{
		inline int min(int a, int b) __GPU_ONLY
		{
			return direct3d::imin(a, b);
		}
		inline int max(int a, int b) __GPU_ONLY
		{
			return direct3d::imax(a, b);
		}
		inline unsigned int min(unsigned int a, unsigned int b) __GPU_ONLY
		{
			return direct3d::umin(a, b);
		}
		inline unsigned int max(unsigned int a, unsigned int b) __GPU_ONLY
		{
			return direct3d::umax(a, b);
		}
	}

	namespace precise_math {

		using direct3d::abs;
		using direct3d::min;
		using direct3d::max;

		inline float abs(float _X) __GPU_ONLY
		{
			return __dp_math_fabsf(_X);
		}
		inline double abs(double _X) __GPU_ONLY
		{
			return __dp_math_fabs(_X);
		}
		inline float min(float a, float b) __GPU_ONLY
		{
			return fmin(a, b);
		}
		inline float max(float a, float b) __GPU_ONLY
		{
			return fmax(a, b);
		}
		inline double min(double a, double b) __GPU_ONLY
		{
			return fmin(a, b);
		}
		inline double max(double a, double b) __GPU_ONLY
		{
			return fmax(a, b);
		}		

		// CPU ONLY CODE PASS
		using std::min;
		using std::max;
		using std::abs;

		inline void sincos
			(
				float  Value,
				float* pSin,
				float* pCos
				)
		{
			assert(pSin);
			assert(pCos);

			// Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
			float quotient = XM_1DIV2PI*Value;
			if (Value >= 0.0f)
			{
				quotient = (float)((int)(quotient + 0.5f));
			}
			else
			{
				quotient = (float)((int)(quotient - 0.5f));
			}
			float y = Value - XM_2PI*quotient;

			// Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
			float sign;
			if (y > XM_PIDIV2)
			{
				y = XM_PI - y;
				sign = -1.0f;
			}
			else if (y < -XM_PIDIV2)
			{
				y = -XM_PI - y;
				sign = -1.0f;
			}
			else
			{
				sign = +1.0f;
			}

			float y2 = y * y;

			// 11-degree minimax approximation
			*pSin = (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f) * y2 + 0.0083333310f) * y2 - 0.16666667f) * y2 + 1.0f) * y;

			// 10-degree minimax approximation
			float p = ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f) * y2 + 0.041666638f) * y2 - 0.5f) * y2 + 1.0f;
			*pCos = sign*p;
		}

	}
	namespace fast_math
	{
		using direct3d::abs;
		using direct3d::min;
		using direct3d::max;

		inline float abs(float _X) __GPU_ONLY
		{
			return __dp_d3d_absf(_X);
		}
		inline float min(float a, float b) __GPU_ONLY
		{
			return fmin(a, b);
		}
		inline float max(float a, float b) __GPU_ONLY
		{
			return fmax(a, b);
		}
		// CPU ONLY CODE PASS
		using std::min;
		using std::max;
		using std::abs;

		_Use_decl_annotations_
			inline void sincos
			(
				float  Value,
				float* pSin,
				float* pCos
				) __CPU_ONLY
		{
			assert(pSin);
			assert(pCos);

			// Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
			float quotient = XM_1DIV2PI*Value;
			if (Value >= 0.0f)
			{
				quotient = (float)((int)(quotient + 0.5f));
			}
			else
			{
				quotient = (float)((int)(quotient - 0.5f));
			}
			float y = Value - XM_2PI*quotient;

			// Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
			float sign;
			if (y > XM_PIDIV2)
			{
				y = XM_PI - y;
				sign = -1.0f;
			}
			else if (y < -XM_PIDIV2)
			{
				y = -XM_PI - y;
				sign = -1.0f;
			}
			else
			{
				sign = +1.0f;
			}

			float y2 = y * y;

			// 7-degree minimax approximation
			*pSin = (((-0.00018524670f * y2 + 0.0083139502f) * y2 - 0.16665852f) * y2 + 1.0f) * y;

			// 6-degree minimax approximation
			float p = ((-0.0012712436f * y2 + 0.041493919f) * y2 - 0.49992746f) * y2 + 1.0f;
			*pCos = sign*p;
		}

	}


	namespace graphics
	{
		namespace direct3d
		{

			inline float dot(float v0, float v1) __GPU
			{
				return v0 * v1;
			}

			inline float dot(float_2 v0, float_2 v1) __GPU
			{
				return v0.x * v1.x + v0.y * v1.y;
			}

			inline float dot(float_3 v0, float_3 v1) __GPU
			{
				return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
			}

			inline float dot(float_4 v0, float_4 v1) __GPU
			{
				return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
			}

			inline float_3 cross(float_3 V1, float_3 V2) __GPU
			{
				float_3 ret;
				ret.x = V1.y*V2.z - V1.z*V2.y;
				ret.y = V1.z*V2.x - V1.x*V2.z;
				ret.z = V1.x*V2.y - V1.y*V2.x;
				return ret;
			}
		}


#define __GPU_INTRINSIC_SIM __GPU

#define __XX_SINCOSE_COMPOENT(comp) \
		__MATH_NS sincos(v.comp, &sf, &cf); \
		s.comp = sf; c.comp = cf;

#define __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_2(T,Func) \
		inline T##_2 Func(T##_2 v0, T##_2 v1) __GPU_INTRINSIC_SIM \
		{ \
			T##_2 ret; \
			ret.x = __MATH_NS Func(v0.x, v1.x);\
			ret.y = __MATH_NS Func(v0.y, v1.y);\
		}
#define __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_3(T,Func) \
		inline T##_3 Func(T##_3 v0, T##_3 v1) __GPU_INTRINSIC_SIM \
		{ \
			T##_3 ret; \
			ret.x = __MATH_NS Func(v0.x, v1.x);\
			ret.y = __MATH_NS Func(v0.y, v1.y);\
			ret.z = __MATH_NS Func(v0.z, v1.z);\
		}
#define __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_4(T,Func) \
		inline T##_4 Func(T##_4 v0, T##_4 v1) __GPU_INTRINSIC_SIM \
		{ \
			T##_4 ret; \
			ret.x = __MATH_NS Func(v0.x, v1.x);\
			ret.y = __MATH_NS Func(v0.y, v1.y);\
			ret.z = __MATH_NS Func(v0.z, v1.z);\
			ret.w = __MATH_NS Func(v0.w, v1.w);\
		}
#define __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOADS(T, Func)\
		__XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_2(T, Func)\
		__XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_3(T, Func)\
		__XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_4(T, Func)


#define __XX_TEMPLATE_COMPOENTS_OVERLOAD_2(T,Func) \
		inline T##_2 Func(T##_2 v) __GPU_INTRINSIC_SIM \
		{ \
			T##_2 ret; \
			ret.x = __MATH_NS Func(v.x);\
			ret.y = __MATH_NS Func(v.y);\
		}
#define __XX_TEMPLATE_COMPOENTS_OVERLOAD_3(T,Func) \
		inline T##_3 Func(T##_3 v) __GPU_INTRINSIC_SIM \
		{ \
			T##_3 ret; \
			ret.x = __MATH_NS Func(v.x);\
			ret.y = __MATH_NS Func(v.y);\
			ret.z = __MATH_NS Func(v.z);\
		}
#define __XX_TEMPLATE_COMPOENTS_OVERLOAD_4(T,Func) \
		inline T##_4 Func(T##_4 v) __GPU_INTRINSIC_SIM \
		{ \
			T##_4 ret; \
			ret.x = __MATH_NS Func(v.x);\
			ret.y = __MATH_NS Func(v.y);\
			ret.z = __MATH_NS Func(v.z);\
			ret.w = __MATH_NS Func(v.w);\
		}

#define __XX_TEMPLATE_COMPOENTS_OVERLOADS(T, Func)\
		__XX_TEMPLATE_COMPOENTS_OVERLOAD_2(T, Func)\
		__XX_TEMPLATE_COMPOENTS_OVERLOAD_3(T, Func)\
		__XX_TEMPLATE_COMPOENTS_OVERLOAD_4(T, Func)


#undef  __MATH_NS
#define __MATH_NS concurrency::direct3d::
#undef  __GPU_INTRINSIC_SIM
#define __GPU_INTRINSIC_SIM __GPU_ONLY

		// overloads for radians/rcp/noise/saturate
		namespace direct3d
		{
#include "amp_hlsl_intrinsic_detail_macro_dx.h"
		}

#undef  __GPU_INTRINSIC_SIM
#define __GPU_INTRINSIC_SIM __GPU

#undef __MATH_NS
#define __MATH_NS concurrency::fast_math::
		namespace fast_math
		{
			using direct3d::dot;
			using direct3d::cross;
			using direct3d::abs;

#include "amp_hlsl_intrinsic_detail_macro.h"
		}

#undef  __MATH_NS
#define __MATH_NS concurrency::precise_math::
		namespace precise_math
		{
			using direct3d::dot;
			using direct3d::cross;
			using direct3d::abs;

#include "amp_hlsl_intrinsic_detail_macro.h"
		}

	}

	namespace hlsl
	{
		using namespace concurrency::graphics;
		using namespace concurrency::direct3d;
		using namespace concurrency::graphics::direct3d;
		using namespace concurrency::fast_math;
		using namespace concurrency::graphics::fast_math;
	}

	namespace hlsl_precise_math
	{
		using namespace concurrency::graphics;
		using namespace concurrency::direct3d;
		using namespace concurrency::graphics::direct3d;
		using namespace concurrency::precise_math;
		using namespace concurrency::graphics::precise_math;
	}

}

#undef __GPU_INTRINSIC_SIM
#undef __MATH_NS
#undef __XX_TEMPLATE_COMPOENTS_OVERLOADS
#undef __XX_TEMPLATE_COMPOENTS_OVERLOAD_4
#undef __XX_TEMPLATE_COMPOENTS_OVERLOAD_3
#undef __XX_TEMPLATE_COMPOENTS_OVERLOAD_2
#undef __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOADS
#undef __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_2
#undef __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_3
#undef __XX_TEMPLATE_BINARY_COMPOENTS_OVERLOAD_4
