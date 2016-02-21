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
 * gemv.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

//-------------------------------------------------------------------------
// GEMV
//-------------------------------------------------------------------------

template <enum class transpose transa, typename value_type, typename x_vector_type, typename y_vector_type>
void gemv(const concurrency::accelerator_view& av, value_type alpha, const concurrency::array_view<const value_type,2>& a, const x_vector_type& x, value_type beta, y_vector_type& y)
{
	concurrency::parallel_for_each(
        av,
        y.extent, 
        [=] (concurrency::index<1> y_idx) restrict(amp)
        {
            value_type result = value_type();
        
            for (int n = 0; n < x.extent[0]; ++n)
            {
                concurrency::index<2> a_idx = (transa == transpose::no_trans ? concurrency::index<2>(n, y_idx[0]) : concurrency::index<2>(y_idx[0], n));
			    concurrency::index<1> x_idx(n);

                auto a_value = a[a_idx];
                if (transa == transpose::conj_trans)
                    a_value = conjugate::op(a_value);
            
                result += a_value * x[x_idx];
            }

            y[y_idx] = alpha * result + beta * y[y_idx];
        }
    );
}
} // namespace _detail

template <typename value_type, typename x_vector_type, typename y_vector_type>
void gemv(const concurrency::accelerator_view& av, enum class transpose transa, value_type alpha, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, value_type beta, y_vector_type& y)
{
    if (transa == transpose::no_trans)
    {
        _detail::gemv<transpose::no_trans>(av, alpha, a, x, beta, y);
    }
    else if (transa == transpose::trans)
    {
        _detail::gemv<transpose::trans>(av, alpha, a, x, beta, y);
    }
    else if (transa == transpose::conj_trans)
    {
        _detail::gemv<transpose::conj_trans>(av, alpha, a, x, beta, y);
    }
}

} // namespace ampblas