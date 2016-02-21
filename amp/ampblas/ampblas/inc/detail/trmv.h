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
 * trmv.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

//-------------------------------------------------------------------------
// TRMV
//  This routine has limited parallelism and should only be used as a
//  building block to enable larger routines.
//-------------------------------------------------------------------------

template <typename value_type, typename x_vector_type, typename y_vector_type> 
void trmv_l(const concurrency::accelerator_view& av, enum class transpose transa, enum class diag diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, y_vector_type& y)
{
    concurrency::parallel_for_each(av, y.extent, [=] (concurrency::index<1> y_idx) restrict(amp)
    {
        value_type result = value_type();

        for (int i=0; i<=y_idx[0]; i++)
        {
            concurrency::index<2> a_idx = (transa == transpose::no_trans ? concurrency::index<2>(i, y_idx[0]) : concurrency::index<2>(y_idx[0], i));
			concurrency::index<1> x_idx(i);

            result += (diag == diag::unit && _detail::is_diag(a_idx) ? x[x_idx] : a[a_idx] * x[x_idx]);
        }

        y[y_idx] = result;
    });
}

template <typename value_type, typename x_vector_type, typename y_vector_type> 
void trmv_u(const concurrency::accelerator_view& av, enum class transpose transa, enum class diag diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, y_vector_type& y)
{
    int n = y.extent[0];

    concurrency::parallel_for_each(av, y.extent, [=] (concurrency::index<1> y_idx) restrict(amp)
    {
        value_type result = value_type();
       
        for (int i=y_idx[0]; i<n; i++)
        {
            concurrency::index<2> a_idx = (transa == transpose::no_trans ? concurrency::index<2>(i, y_idx[0]) : concurrency::index<2>(y_idx[0], i));
			concurrency::index<1> x_idx(i);

            result += (diag == diag::unit && _detail::is_diag(a_idx) ? x[x_idx] : a[a_idx] * x[x_idx]);
        }

        y[y_idx] = result;
    });
}

} // namespace _detail

template <typename value_type, typename x_vector_type, typename y_vector_type> 
void trmv(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose transa, enum class diag diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x, y_vector_type& y)
{
    if ((uplo == uplo::lower) ^ (transa != transpose::no_trans))
	    _detail::trmv_l(av, transa, diag, a, x, y);
    else
        _detail::trmv_u(av, transa, diag, a, x, y);
}

} // namespace ampblas
