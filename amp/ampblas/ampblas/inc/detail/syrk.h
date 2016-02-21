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
 * syrk.h
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_SYRK_H
#define AMPBLAS_SYRK_H

#include "ampblas_dev.h"

#include "gemm.h"

namespace ampblas {
namespace _detail {

template <typename trans_op, int tile_size, typename scalar_type, typename a_type, typename c_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, scalar_type alpha, const a_type& a_mat, scalar_type beta, const c_type& c_mat)
{
    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<c_type>::value, "c_type must be an array_view" ); 

    // 
    typedef c_type::value_type value_type;

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    const int n = c_mat.extent[0];
    const int k = (trans == transpose::no_trans ? a_mat.extent[0] : a_mat.extent[1]);
    const int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles,tile_size*tiles);

    concurrency::parallel_for_each(
        av,
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> idx_c) restrict(amp)
        {
            tile_static value_type at[tile_size+1][tile_size]; // "a" tile
            tile_static value_type att[tile_size+1][tile_size]; // "a" transpose tile

            auto i = idx_c.local[1];
            auto j = idx_c.local[0];
            auto tile_i = idx_c.tile[1];
            auto tile_j = idx_c.tile[0];
            auto tile_i_origin = idx_c.tile_origin[1];
            auto tile_j_origin = idx_c.tile_origin[0];
            auto global_i = idx_c.global[1];
            auto global_j = idx_c.global[0];

            // quick return path for unnecessary tiles
            // skips too early for operations with just a beta component, but those shouldn't be handled by this routine
            if ( (uplo==uplo::upper && tile_j < tile_i) || (uplo==uplo::lower && tile_i < tile_j) ) 
                return;

            bool notrans = trans == transpose::no_trans;
            value_type& at_local = notrans ? at[i][j] : at[j][i];
            value_type& att_local = notrans ? att[i][j] : att[j][i];

            value_type out = value_type(0);
            for ( auto ii=0; ii < k; ii += tile_size )
            {
                auto a_idx = notrans ? concurrency::index<2>(i+ii, tile_i_origin+j) : concurrency::index<2>(i+tile_i_origin, ii+j);
                auto at_idx = notrans ? concurrency::index<2>(i+ii, tile_j_origin+j) : concurrency::index<2>(i+tile_j_origin, ii+j);
                auto v = _detail::guarded_read<true>(a_mat,a_idx);

                at_local = v;
                att_local = tile_i==tile_j ? v : (_detail::guarded_read<true>(a_mat,at_idx));

                // apply transpose operation
                if (trans == transpose::no_trans)
                    att_local = trans_op::op(att_local);
                else
                    at_local = trans_op::op(at_local);

                idx_c.barrier.wait_with_tile_static_memory_fence();

                // multiply matrices
                int end = _detail::min(tile_size,k-ii);
                for ( auto kk=0; kk<end; ++kk )
                    out += alpha*at[kk][i]*att[kk][j];

                idx_c.barrier.wait_with_tile_static_memory_fence();
            }
            if ( (uplo==uplo::upper && global_j >= global_i) || (uplo==uplo::lower && global_i >= global_j) && global_i<n && global_j<n )
            {
                auto c_val = c_mat[idx_c];

                if ( global_i == global_j )
                {
                    _detail::only_real(c_val);
                }

                if ( beta != scalar_type() )
                    out += beta*c_val;

                c_mat[idx_c] = out;
            }
        }
    );
}

// tuning interface
template <int rb, typename trans_op, typename scalar_type, typename a_type, typename c_type>
void recursive_syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, int n, int k, scalar_type alpha, const a_type& a, scalar_type beta, const c_type& c)
{
    const order S = order::col_major;
    typedef typename a_type::value_type value_type;

    int n1, n2;
    // 't' for noop and 'c' for complex conjugate
    const enum class transpose trans_type = transpose_type<trans_op>::value;

    if( ( n1 = n - rb ) <= 0 )
    {
        int a_row = (trans == transpose::no_trans ? n : k);
        int a_col = (trans == transpose::no_trans ? k : n); 

        // tuning interface
        _detail::syrk<trans_op>( av, uplo, trans, alpha, a.section(extent<S>(a_row, a_col)), beta, c.section(extent<S>(n,n)) );
        return;
    }

    n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

    if ( uplo == uplo::upper && trans != transpose::no_trans )
    {
        recursive_syrk<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, beta, c );
        gemm( av, trans_type, transpose::no_trans, n1, n2, k, value_type(alpha), a, a.section(index<S>(0,n1)), value_type(beta), c.section(index<S>(0,n1)) );
        recursive_syrk<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(0,n1)), beta, c.section(index<S>(n1,n1)) );
    }
    else if ( uplo == uplo::upper && trans == transpose::no_trans )
    {
        recursive_syrk<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, beta, c );
        gemm( av, transpose::no_trans, trans_type, n1, n2, k, value_type(alpha), a, a.section(index<S>(n1,0)), value_type(beta), c.section(index<S>(0,n1)) );
        recursive_syrk<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(n1,0)), beta, c.section(index<S>(n1,n1)) );
    }
    else if ( uplo == uplo::lower && trans != transpose::no_trans )
    {
        recursive_syrk<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, beta, c );
        gemm( av, trans_type, transpose::no_trans, n2, n1, k, value_type(alpha), a.section(index<S>(0,n1)), a, value_type(beta), c.section(index<S>(n1,0)) );
        recursive_syrk<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(0,n1)), beta, c.section(index<S>(n1,n1)) );
    }
    else if ( uplo == uplo::lower && trans == transpose::no_trans )
    {
        recursive_syrk<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, beta, c );
        gemm( av, transpose::no_trans, trans_type, n2, n1, k, value_type(alpha), a.section(index<S>(n1,0)), a, value_type(beta), c.section(index<S>(n1,0)) );
        recursive_syrk<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(n1,0)), beta, c.section(index<S>(n1,n1)) );
    }
}

// tuning interface
template <typename trans_op, typename scalar_type, typename a_type, typename c_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, scalar_type alpha, const a_type& a_mat, scalar_type beta, const c_type& c_mat)
{
    // tuning parameters
    const int tile_size = 16;

    // main routine
    syrk<trans_op, tile_size>(av, uplo, trans, alpha, a_mat, beta, c_mat);
}

} // namespace _detail

template <typename trans_op, typename scalar_type, typename a_type, typename c_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, scalar_type alpha, const a_type& a, scalar_type beta, const c_type& c)
{
    const order S = order::col_major;

    const int n = _detail::rows<S>(c.extent);
    const int a_row = _detail::rows<S>(a.extent);
    const int a_col = _detail::columns<S>(a.extent);
    const int k = (trans == transpose::no_trans ? a_col : a_row);

    // recursive block size
    const int rb = 1024;

    // forward to recursive interface
    _detail::recursive_syrk<rb, trans_op>(av, uplo, trans, n, k, alpha, a, beta, c);
}

// implied noop interface
template <typename scalar_type, typename a_type, typename c_type>
void syrk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, scalar_type alpha, const a_type& a_mat, scalar_type beta, const c_type& c_mat)
{
    syrk<_detail::noop>(av, uplo, trans, alpha, a_mat, beta, c_mat);
}

// implied conjugate operation interface
template <typename scalar_type, typename a_type, typename c_type>
void herk(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, scalar_type alpha, const a_type& a_mat, scalar_type beta, const c_type& c_mat)
{
    syrk<_detail::conjugate>(av, uplo, trans, alpha, a_mat, beta, c_mat);
}

} // namespace ampblas


#endif // AMPBLAS_SYRK_H 
