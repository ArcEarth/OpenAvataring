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
 * trsv.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename value_type, typename x_vector_type> 
void trsv_l(const concurrency::accelerator_view& av, enum class transpose transa, const enum class diag diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x)
{
    const int n = x.extent[0];

    concurrency::parallel_for_each(av, make_extent(tile_size).tile<tile_size>(), [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        for (int j=0; j<n; j++)
        {
            if (diag == diag::non_unit)
            {
                if (tid.local[0] == 0)
                    x[concurrency::index<1>(j)] /= a[concurrency::index<2>(j,j)];

                tid.barrier.wait();
            }

            value_type alpha = x[concurrency::index<1>(j)];

            for (int i=tid.local[0]+j+1; i<n; i+=tile_size)
                x[concurrency::index<1>(i)] -= alpha * a[transa == transpose::no_trans ? concurrency::index<2>(j,i) : concurrency::index<2>(i,j)];

            tid.barrier.wait();
        }
    });
}

template <int tile_size, typename value_type, typename x_vector_type> 
void trsv_u(const concurrency::accelerator_view& av, enum class transpose transa, const enum class diag diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x)
{
    int n = x.extent[0];

    // compiler work around
    const int dummy = 1;

    concurrency::parallel_for_each(av, make_extent(tile_size).tile<tile_size>(), [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        for (int jj=n-1; jj>=0; jj--)
        {
            // compiler work around
            int j = dummy ? jj : 0;

            if (diag == diag::non_unit)
            {
                if (tid.local[0] == 0)
                    x[concurrency::index<1>(j)] /= a[concurrency::index<2>(j,j)];

                tid.barrier.wait();
            }

            value_type alpha = x[concurrency::index<1>(j)];

            for (int i=tid.local[0]; i<j; i+=tile_size)
                x[concurrency::index<1>(i)] -= alpha * a[transa == transpose::no_trans ? concurrency::index<2>(j,i) : concurrency::index<2>(i,j)];
            tid.barrier.wait();
        }
    });
}

} // namespace _detail

//-------------------------------------------------------------------------
// TRSV
//  The current implementation of this routine has minimial parallelism and
//  should only be used as a building block to enable larger routines.
//-------------------------------------------------------------------------

template <typename value_type, typename x_vector_type> 
void trsv(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose transa, enum class diag diag, const concurrency::array_view<const value_type,2>& a, x_vector_type& x)
{
    // tuning parameters
    const int tile_size = 256;

    if ((uplo == uplo::lower) ^ (transa != transpose::no_trans))
    {
        // lower + no trans <==> upper + trans
        _detail::trsv_l<tile_size>(av, transa, diag, a, x);
    }
    else
    {
        // upper + no trans <==> lower + trans
        _detail::trsv_u<tile_size>(av, transa, diag, a, x);
    }
}

} // namespace ampblas
