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
 * algorithm.h 
 *
 * Common generic matrix algorithms
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_UTILITY_ALGORITHM_H
#define AMPBLAS_UTILITY_ALGORITHM_H

#include "ampblas_config.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// generic fill algorithm on any multi-dimensional container
template <int rank, typename value_type, typename x_type>
void fill(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, value_type&& value, x_type&& x)
{
    concurrency::parallel_for_each(av, e, [=] (concurrency::index<rank> idx) restrict(amp) 
    {
        x[idx] = value;
    });
}

// triangular fill for rank-2 containers
template <typename value_type>
void fill(const concurrency::accelerator_view& av, enum class uplo uplo, const value_type& value, const concurrency::array_view<value_type,2>& x)
{
    if (uplo == uplo::upper)
    {
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i <= j )
                x[idx] = value;
        });
    }
    else
    {
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i >= j )
                x[idx] = value;
        });
    }
}

// generic scale algorithm on any multi-dimensional container
template <int rank, typename value_type, typename x_type>
void scale(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, const value_type& value, x_type&& x)
{
    concurrency::parallel_for_each(
        av,
        e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            x[idx] *= value;
        }
    );
}

// generic swap algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
void swap(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, x_type&& x, y_type&& y)
{
    concurrency::parallel_for_each(
        av, 
        e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            auto tmp = y[idx];
            y[idx] = x[idx];
            x[idx] = tmp;
        }
    );
}

// triangular fill for rank-2 containers
template <typename scalar_type, typename value_type>
void scale(const concurrency::accelerator_view& av, enum class uplo uplo, const scalar_type& value, const concurrency::array_view<value_type,2>& x)
{
    if ( uplo == uplo::upper )
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i <= j )
                x[idx] = value*x[idx];
        });
    else
        concurrency::parallel_for_each(av, x.extent, [=] (concurrency::index<2> idx) restrict(amp) 
        {
            auto i = idx[1];
            auto j = idx[0];
            if ( i >= j )
                x[idx] = value*x[idx];
        });
}

// generic copy algorithm on any multi-dimensional container
template <int rank, typename x_type, typename y_type>
void copy(const concurrency::accelerator_view& av, const concurrency::extent<rank>& e, x_type&& x, y_type&& y)
{
    concurrency::parallel_for_each(av, e, 
        [=] (concurrency::index<rank> idx) restrict(amp) 
        {
            y[idx] = x[idx];
        }
    );
}

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_UTILITY_ALGORITHM_H