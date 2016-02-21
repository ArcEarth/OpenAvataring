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
 * syr2k.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

#include "gemm.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename trans_op, typename alpha_type, typename beta_type, typename a_type, typename b_type, typename c_type>
void syr2k(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, alpha_type alpha, const a_type& a_mat, const b_type& b_mat, beta_type beta, const c_type& c_mat)
{
    const enum class order S = order::col_major;

    typedef typename c_type::value_type value_type;

    const int n = rows<S>(c_mat.extent); 
    const int k = (trans == transpose::no_trans ? columns<S>(a_mat.extent) : rows<S>(a_mat.extent));

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles, tile_size*tiles);

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
            if ( (uplo==uplo::upper && tile_j < tile_i) || (uplo==uplo::lower && tile_i < tile_j) ) return;

            bool notrans = trans == transpose::no_trans;
            value_type& at_local = notrans ? at[i][j] : at[j][i];
            value_type& att_local = notrans ? att[i][j] : att[j][i];

            value_type out=value_type(0);
            for ( auto ii=0; ii < k; ii += tile_size )
            {
                auto a_idx = notrans?concurrency::index<2>(i+ii, tile_i_origin+j):concurrency::index<2>(tile_i_origin+i, ii+j);
                auto bt_idx = notrans?concurrency::index<2>(i+ii, tile_j_origin+j):concurrency::index<2>(tile_j_origin+i, ii+j);

                at_local  = _detail::guarded_read<true>(a_mat,a_idx);
                att_local = _detail::guarded_read<true>(b_mat,bt_idx);

                // apply transpose operation
                att_local = trans_op::op(att_local);

                idx_c.barrier.wait_with_tile_static_memory_fence();

                // multiply matrices
                int end = _detail::min(tile_size,k-ii);
                if ( tile_i == tile_j ) // shortcut for diagonal tiles
                {
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*(at[kk][i]*att[kk][j]+at[kk][j]*att[kk][i]);
                    }
                }
                else
                {
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*at[kk][i]*att[kk][j];
                    }
                }

                idx_c.barrier.wait_with_tile_static_memory_fence();
                if ( tile_i == tile_j ) 
                    continue; // diagonal tiles skip some memory time

                // swap matrices, repeat
                at_local  = _detail::guarded_read<true>(b_mat,a_idx);
                att_local = _detail::guarded_read<true>(a_mat,bt_idx);

                // apply transpose operation
                att_local = trans_op::op(att_local);

                idx_c.barrier.wait_with_tile_static_memory_fence();

                // multiply matrices
                for ( auto kk=0; kk<end; ++kk )
                {
                    out += alpha*at[kk][i]*att[kk][j];
                }
            }
            if ( (uplo==uplo::upper && global_j >= global_i) || (uplo==uplo::lower && global_i >= global_j) && global_i<n && global_j<n )
            {
                if ( beta != beta_type() )
                    out += beta*c_mat[idx_c];
                c_mat[idx_c] = out;
            }
        }
    );
}

template <int rb, typename trans_op, typename alpha_type, typename beta_type, typename a_type, typename b_type, typename c_type>
void recursive_syr2k(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, int n, int k, alpha_type alpha, const a_type& a, const b_type& b, beta_type beta, const c_type& c )
{
    const order S = order::col_major;
    typedef typename c_type::value_type value_type;

    int n1, n2;

    // if trans_op == noop : trans_type = 'T'
    // if trans_op == conj : trans_type = 'C'
    const enum class transpose trans_type = transpose_type<trans_op>::value;

    // ln case
    if ( uplo == uplo::lower && trans == transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            _detail::syr2k<trans_op>( av, uplo, trans, alpha, a.section(extent<S>(n,k)), b.section(extent<S>(n,k)), value_type(beta), c.section(extent<S>(n,n)) );
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb ); 

        recursive_syr2k<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, b, beta, c );
        gemm( av, transpose::no_trans, trans_type, n2, n1, k, value_type(alpha), a.section(index<S>(n1,0)), b, value_type(beta), c.section(index<S>(n1,0)) );
        gemm( av, transpose::no_trans, trans_type, n2, n1, k, value_type(alpha), b.section(index<S>(n1,0)), a, value_type(1), c.section(index<S>(n1,0)) );
        recursive_syr2k<rb, trans_op>( av, uplo, trans, n2, k, alpha,  a.section(index<S>(n1,0)), b.section(index<S>(n1,0)), beta, c.section(index<S>(n1,n1)) );
    }

    // lt case
    if ( uplo == uplo::lower && trans != transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            _detail::syr2k<trans_op>( av, uplo, trans, alpha, a.section(extent<S>(k,n)), b.section(extent<S>(k,n)), value_type(beta), c.section(extent<S>(n,n)) );
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_syr2k<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, b, beta, c );
        gemm( av, trans_type, transpose::no_trans, n2, n1, k, value_type(alpha), a.section(index<S>(0,n1)), b, value_type(beta), c.section(index<S>(n1,0)) );
        gemm( av, trans_type, transpose::no_trans, n2, n1, k, value_type(alpha), b.section(index<S>(0,n1)), a, value_type(1), c.section(index<S>(n1,0)) );
        recursive_syr2k<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(0,n1)), b.section(index<S>(0,n1)), beta, c.section(index<S>(n1,n1)) );
    }

    // un case
    if ( uplo == uplo::upper && trans == transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            _detail::syr2k<trans_op>( av, uplo, trans, alpha, a.section(extent<S>(n,k)), b.section(extent<S>(n,k)), value_type(beta), c.section(extent<S>(n,n)) );
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_syr2k<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, b, beta, c );
        gemm( av, transpose::no_trans, trans_type, n1, n2, k, value_type(alpha), a, b.section(index<S>(n1,0)), value_type(beta), c.section(index<S>(0,n1)) );
        gemm( av, transpose::no_trans, trans_type, n1, n2, k, value_type(alpha), b, a.section(index<S>(n1,0)), value_type(1), c.section(index<S>(0,n1)) );
        recursive_syr2k<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(n1,0)), b.section(index<S>(n1,0)), beta, c.section(index<S>(n1,n1)) );

    }

    // ut case
    if ( uplo == uplo::upper && trans != transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            _detail::syr2k<trans_op>( av, uplo, trans, alpha, a.section(extent<S>(k,n)), b.section(extent<S>(k,n)), value_type(beta), c.section(extent<S>(n,n)) );
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_syr2k<rb, trans_op>( av, uplo, trans, n1, k, alpha, a, b, beta, c );
        gemm( av, trans_type, transpose::no_trans, n1, n2, k, value_type(alpha), a, b.section(index<S>(0,n1)), value_type(beta), c.section(index<S>(0,n1)) );
        gemm( av, trans_type, transpose::no_trans, n1, n2, k, value_type(alpha), b, a.section(index<S>(0,n1)), value_type(1), c.section(index<S>(0,n1)) );
        recursive_syr2k<rb, trans_op>( av, uplo, trans, n2, k, alpha, a.section(index<S>(0,n1)), b.section(index<S>(0,n1)), beta, c.section(index<S>(n1,n1)) );
    }
}

template <typename trans_op, typename alpha_type, typename beta_type, typename a_type, typename b_type, typename c_type>
void syr2k(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, alpha_type alpha, const a_type& a_mat, const b_type& b_mat, beta_type beta, const c_type& c_mat)
{
    // tuning parameters
    const int tile_size = 16;

    // call main routine
    _detail::syr2k<tile_size,trans_op>(av, uplo, trans, alpha, a_mat, b_mat, beta, c_mat);
}

} // namespace _detail


template <typename trans_op, typename alpha_type, typename beta_type, typename a_type, typename b_type, typename c_type>
void syr2k(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, alpha_type alpha, const a_type& a, const b_type& b, beta_type beta, const c_type& c)
{
    // extract sizes
    const order S = order::col_major;
    const int n = _detail::rows<S>(c.extent); 
    const int k = (trans == transpose::no_trans ? _detail::columns<S>(a.extent) : _detail::rows<S>(a.extent));

    // recursive block size
    const int rb = 1024;

    // pass to recursive routine
    _detail::recursive_syr2k<rb, trans_op>(av, uplo, trans, n, k, alpha, a, b, beta, c);
}

template <typename trans_op, typename alpha_type, typename beta_type, typename a_type, typename b_type, typename c_type>
void syr2k(const concurrency::accelerator_view& av, enum class uplo uplo, enum class transpose trans, int n, int k, alpha_type alpha, const a_type& a, const b_type& b, beta_type beta, const c_type& c)
{
    // build sizes
    const order S = order::col_major;
    concurrency::extent<2> c_extent = _detail::extent<S>(n,n);
    concurrency::extent<2> ab_extent = (trans == transpose::no_trans ? _detail::extent<S>(n,k) : _detail::extent<S>(k,n));

    // forward to "full" routine
    syr2k<trans_op>( av, uplo, trans, alpha, a.section(ab_extent), b.section(ab_extent), beta, c.section(c_extent) );
}

} // namespace ampblas
