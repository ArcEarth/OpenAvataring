#pragma once
#include <amp_short_vectors.h>
#include <amp_math.h>

#define __Out_Param__(T,name) T& name
#define __MATH_NS concurrency::fast_math::


namespace Concurrency
{
	namespace graphics
	{

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

#define __XX_TEMPLATE_LENGTH(T) \
		inline T length(T V) __GPU \
		{ \
			return __MATH_NS sqrt(dot(V,V)); \
		}

		__XX_TEMPLATE_LENGTH(float_2)
			__XX_TEMPLATE_LENGTH(float_3)
			__XX_TEMPLATE_LENGTH(float_4)

#undef __XX_TEMPLATE_LENGTH

#define __XX_TEMPLATE_NORMALIZE(T) \
		inline T normalize(T V) __GPU \
		{ \
			T ret; \
			ret /= __MATH_NS sqrt(dot(V,V)); \
			return ret;\
		}

			__XX_TEMPLATE_NORMALIZE(float_2)
			__XX_TEMPLATE_NORMALIZE(float_3)
			__XX_TEMPLATE_NORMALIZE(float_4)

#undef __XX_TEMPLATE_NORMALIZE

		inline void sincos(float v, __Out_Param__(float, s), __Out_Param__(float, c) ) __GPU_ONLY
		{
			__MATH_NS sincos(v, &s, &c);
		}

#define __XX_SINCOSE_COMPOENT(comp) \
		__MATH_NS sincos(v.comp, &sf, &cf); \
		s.comp = sf; c.comp = cf;

		inline void sincos(float_2 v, __Out_Param__(float_2, s), __Out_Param__(float_2, c)) __GPU_ONLY
		{
			float sf, cf;
			__XX_SINCOSE_COMPOENT(x);
			__XX_SINCOSE_COMPOENT(y);
		}

		inline void sincos(float_3 v, __Out_Param__(float_3, s), __Out_Param__(float_3, c)) __GPU_ONLY
		{
			float sf, cf;
			__XX_SINCOSE_COMPOENT(x);
			__XX_SINCOSE_COMPOENT(y);
			__XX_SINCOSE_COMPOENT(z);
		}

		inline void sincos(float_4 v, __Out_Param__(float_4, s), __Out_Param__(float_4, c)) __GPU_ONLY
		{
			float sf, cf;
			__XX_SINCOSE_COMPOENT(x);
			__XX_SINCOSE_COMPOENT(y);
			__XX_SINCOSE_COMPOENT(z);
			__XX_SINCOSE_COMPOENT(w);
		}

#define __XX_TEMPLATE_COMPOENTS_OVERLOAD_2(T,Func) \
		inline T##_2 Func(T##_2 v) __GPU \
		{ \
			T##_2 ret; \
			ret.x = Func(v.x);\
			ret.y = Func(v.y);\
		}

#define __XX_TEMPLATE_COMPOENTS_OVERLOAD_3(T,Func) \
		inline T##_3 Func(T##_3 v) __GPU \
		{ \
			T##_3 ret; \
			ret.x = Func(v.x);\
			ret.y = Func(v.y);\
			ret.z = Func(v.z);\
		}

#define __XX_TEMPLATE_COMPOENTS_OVERLOAD_4(T,Func) \
		inline T##_4 Func(T##_4 v) __GPU \
		{ \
			T##_4 ret; \
			ret.x = Func(v.x);\
			ret.y = Func(v.y);\
			ret.z = Func(v.z);\
			ret.w = Func(v.w);\
		}

#define __XX_TEMPLATE_COMPOENTS_OVERLOADS (T, Func)\
		__XX_TEMPLATE_COMPOENTS_OVERLOAD_2(T, Func)\
		__XX_TEMPLATE_COMPOENTS_OVERLOAD_3(T, Func)\
		__XX_TEMPLATE_COMPOENTS_OVERLOAD_4(T, Func)

//#define __XX_TEMPLATE_COMPOENTS_OVERLOADS_FLOAT_DOUBLE(Func)  \
//		__XX_TEMPLATE_COMPOENTS_OVERLOADS(float,Func##f) \
//		__XX_TEMPLATE_COMPOENTS_OVERLOADS(double,Func) 
//
//		__XX_TEMPLATE_COMPOENTS_OVERLOADS_FLOAT_DOUBLE(sin)
//		__XX_TEMPLATE_COMPOENTS_OVERLOADS_FLOAT_DOUBLE(cos)

#undef __XX_TEMPLATE_COMPOENTS_OVERLOADS
#undef __XX_TEMPLATE_COMPOENTS_OVERLOAD_4
#undef __XX_TEMPLATE_COMPOENTS_OVERLOAD_3
#undef __XX_TEMPLATE_COMPOENTS_OVERLOAD_2
	}
}