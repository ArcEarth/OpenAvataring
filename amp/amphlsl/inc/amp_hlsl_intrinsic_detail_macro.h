template <typename T>
inline std::enable_if_t<std::is_floating_point<typename short_vector_traits<T>::value_type>::value,
	typename short_vector_traits<T>::value_type>
length(T V) __GPU
{
	return  __MATH_NS sqrt(dot(V, V));
}

template <typename T>
inline std::enable_if_t<std::is_floating_point<typename short_vector_traits<T>::value_type>::value, T>
normalize(T V) __GPU
{
	return  V / __MATH_NS sqrt(dot(V, V));
}

__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, abs)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, sin)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, cos)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, tan)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, acos)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, asin)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, atan)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, sinh)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, cosh)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, tanh)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, round)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, ceil)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, floor)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, trunc)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, sqrt)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, exp)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, exp2)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, log)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, log2)
__XX_TEMPLATE_COMPOENTS_OVERLOADS(float, log10)

__XX_TEMPLATE_BINARY_COMPOENTS_OVERLOADS(float, min)
__XX_TEMPLATE_BINARY_COMPOENTS_OVERLOADS(float, max)

#include "amp_hlsl_intrinsic_sincos.h"