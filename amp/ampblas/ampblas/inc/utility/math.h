/*----------------------------------------------------------------------------
 * Copyright © Microsoft Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not 
 * use this file except in compliance with the License.  You may obtain a copy 
 * of the License at http://www.apache.org/licenses/LICENSE-2.0  
 * 
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED 
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, 
 * MERCHANTABLITY OR NON-INFRINGEMENT. 
 *
 * See the Apache Version 2.0 License for specific language governing 
 * permissions and limitations under the License.
 *---------------------------------------------------------------------------
 *
 * math.h 
 *
 * Commonly used math routines and functors
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_UTILITY_MATH_H
#define AMPBLAS_UTILITY_MATH_H

#include "ampblas_config.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

template <typename T>
inline T abs(const T& val) restrict(cpu, amp)
{
    return val >= 0 ? val: -val;
}

// some routines use this form of abs() for complex values
template <typename value_type>
inline value_type abs_1(const value_type& val) restrict(cpu, amp)
{
    return abs(val);
}

template <typename value_type>
inline value_type abs_1(const complex<value_type>& val) restrict(cpu, amp)
{
    return abs(val.real()) + abs(val.imag());
}

template<typename T>
inline const T& max(const T& a, const T& b) restrict(cpu, amp) 
{
    return a > b ? a : b;
}

template<typename T>
inline const T& min(const T& a, const T& b) restrict(cpu, amp) 
{
    return a < b ? a : b;
}

// returns a value whose absolute value matches that of a, but whose sign bit matches that of b.
template <typename T>
inline T copysign(const T& a, const T& b) restrict(cpu, amp) 
{
    T x = _detail::abs(a);
    return (b >= 0 ? x : -x);
}

template <typename T>
struct maximum
{
    const T& operator()(const T& lhs, const T& rhs) const restrict (cpu, amp) 
    { 
        return _detail::max(lhs, rhs);
    }
};

template <typename T>
struct sum 
{
    T operator()(const T& lhs, const T& rhs) const restrict (cpu, amp) 
    { 
        return lhs + rhs; 
    }
};

template <typename value_type>
struct subtract
{
    value_type subtrahend;

    subtract(const value_type& subtrahend) restrict (cpu,amp)
        : subtrahend(subtrahend) 
    {
    }

    inline void operator()(value_type& minuend) const restrict (cpu,amp)
    {
        minuend -= subtrahend;
    }
};

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_UTILITY_MATH_H