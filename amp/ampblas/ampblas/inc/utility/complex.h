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
 * complex.h 
 *
 * Helper routines related to complex number type; these routines were 
 * considered too speciailized for the normal complex header
 *
 *---------------------------------------------------------------------------*/


#ifndef AMPBLAS_UTILITY_COMPLEX_H
#define AMPBLAS_UTILITY_COMPLEX_H

#include "ampblas_config.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

struct noop
{
    template <typename value_type>
    static inline const value_type op(const value_type& val) restrict (cpu, amp)
    {
        return val;
    }
};

struct conjugate
{
    template <typename value_type>
    static inline const value_type op(const value_type& val) restrict (cpu, amp)
    {
        // noop for real types
        return val;
    }

    template <typename value_type>
    static inline const complex<value_type> op(const complex<value_type>& val) restrict (cpu, amp)
    {
        complex<value_type> ret(val.real(), -val.imag());
        return ret;
    }
};

// transpsoe type helper
template <typename trans_op>
struct transpose_type
{
    static const enum class transpose value = transpose::trans;
};

template <>
struct transpose_type<conjugate>
{
    static const enum class transpose value = transpose::conj_trans;
};

template <typename value_type>
inline void only_real(const value_type&) restrict(cpu, amp)
{
    // noop for real
}

template <typename value_type>
inline void only_real(complex<value_type>& val) restrict(cpu, amp)
{
    val.imag(value_type());
}

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_UTILITY_COMPLEX_H