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
 * symm.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

#include "gemm.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename trans_op, typename scalar_type, typename a_type, typename b_type, typename c_type>
void symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, scalar_type alpha, const a_type& a_mat, const b_type& b_mat, scalar_type beta, c_type& c_mat)
{
    typedef typename c_type::value_type value_type;

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    const enum class order S = order::col_major;
    const int m = rows<S>(c_mat.extent);
    const int n = columns<S>(c_mat.extent);

    int tiles_m = (m+tile_size-1)/tile_size;
    int tiles_n = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles_m,tile_size*tiles_n);

    if ( side == side::left )
        concurrency::parallel_for_each (
            av,
            e.tile<tile_size,tile_size>(),
            [=] (concurrency::tiled_index<tile_size,tile_size> idx_c) restrict(amp)
            {
                tile_static value_type at[tile_size+1][tile_size]; // "a" tile
                tile_static value_type bt[tile_size][tile_size]; // "b" tile

                auto i = idx_c.local[1];
                auto j = idx_c.local[0];
                auto tile_i = idx_c.tile[1];
                auto tile_i_origin = idx_c.tile_origin[1];
                auto global_i = idx_c.global[1];
                auto global_j = idx_c.global[0];

                int tile_origin = 0;
                value_type out=value_type(0);
                for ( int tile=0; tile < tiles_m; ++tile, tile_origin += tile_size )
                {
                    // depending on A's symmetry (uplo), need to adjust load coordinates
                    concurrency::index<2> a_idx;
                    if ( tile_i == tile )
                    {
                        // diagonal tile, need to load half and fill missing symmetry
                        if ( (uplo == uplo::upper && i <= j) || (uplo == uplo::lower && i >= j) )
                        {
                            a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            // data is present - read it
                            auto v = a_mat[a_idx];
                            at[i][j] = v;

                            // fill in the missing other triangle as well
                            if ( i != j ) at[j][i] = v;
                        }
                    }
                    else if ( (uplo == uplo::upper && tile_i < tile) || (uplo == uplo::lower && tile_i > tile) )
                    {
                        // simple case, tile is fully present - read it
                        a_idx = concurrency::index<2>(tile_origin+j,global_i);
                        at[i][j] = a_mat[a_idx];
                    }
                    else
                    {
                        // need to grab the transpose tile - and transpose it as we read
                        a_idx = concurrency::index<2>(tile_i_origin+j,tile_origin+i);
                        at[j][i] = trans_op::op(a_mat[a_idx]);
                    }

                    auto b_idx = concurrency::index<2>(global_j,tile_origin+i);
                    bt[i][j] = b_mat[b_idx];

                    idx_c.barrier.wait();

                    // multiply matrices
                    int end = _detail::min(tile_size,m-tile_origin);
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*at[i][kk]*bt[kk][j];
                    }

                    idx_c.barrier.wait();

                }
                if ( global_i<m && global_j<n )
                {
                    if ( beta != value_type() )
                        out += beta*c_mat[idx_c];
                    c_mat[idx_c] = out;
                }
            }
        );
    else
        concurrency::parallel_for_each (
            av,
            e.tile<tile_size,tile_size>(),
            [=] (concurrency::tiled_index<tile_size,tile_size> idx_c) restrict(amp)
            {
                tile_static value_type at[tile_size+1][tile_size]; // "a" tile
                tile_static value_type bt[tile_size][tile_size]; // "b" tile

                auto i = idx_c.local[1];
                auto j = idx_c.local[0];
                auto tile_j = idx_c.tile[0];
                auto tile_j_origin = idx_c.tile_origin[0];
                auto global_i = idx_c.global[1];
                auto global_j = idx_c.global[0];

                int tile_origin = 0;
                value_type out=value_type(0);
                for ( int tile=0; tile < tiles_n; ++tile, tile_origin += tile_size )
                {
                    // depending on A's symmetry (uplo), need to adjust load coordinates
                    concurrency::index<2> a_idx;
                    if ( tile_j == tile )
                    {
                        // diagonal tile, need to load half and fill missing symmetry
                        if ( (uplo == uplo::upper && i <= j) || (uplo == uplo::lower && i >= j) )
                        {
                            a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            // data is present - read it
                            auto v = a_mat[a_idx];
                            at[i][j] = v;

                            // fill in the missing other triangle as well
                            if ( i != j ) at[j][i] = v;
                        }
                    }
                    else if ( (uplo == uplo::upper && tile_j > tile) || (uplo == uplo::lower && tile_j < tile) )
                    {
                        // simple case, tile is fully present - read it
                        a_idx = concurrency::index<2>(global_j,tile_origin+i);
                        at[i][j] = a_mat[a_idx];
                    }
                    else
                    {
                        // need to grab the transpose tile - and transpose it as we read
                        a_idx = concurrency::index<2>(tile_origin+j,tile_j_origin+i);
                        at[j][i] = trans_op::op(a_mat[a_idx]);
                    }

                    auto b_idx = concurrency::index<2>(tile_origin+j,global_i);
                    bt[i][j] = b_mat[b_idx];

                    idx_c.barrier.wait();

                    // multiply matrices
                    int end = _detail::min(tile_size,n-tile_origin);
                    for ( auto kk=0; kk<end; ++kk )
                    {
                        out += alpha*bt[i][kk]*at[kk][j];
                    }

                    idx_c.barrier.wait();

                }
                if ( global_i<m && global_j<n )
                {
                    if ( beta != value_type() )
                        out += beta*c_mat[idx_c];
                    c_mat[idx_c] = out;
                }
            }
        );
}

template <int rb, typename trans_op, typename scalar_type, typename a_type, typename b_type, typename c_type>
void recursive_symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, int m, int n, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    // only column major support for now
    const enum class order S = order::col_major;

    int m1, n1;
    int m2, n2;

    // if trans_op == noop : trans_type = 'T'
    // if trans_op == conj : trans_type = 'C'
    const enum class transpose trans_type = transpose_type<trans_op>::value;

    scalar_type one = scalar_type(1);

    // ll case
    if ( side == side::left && uplo == uplo::lower )
    {
        if( ( m1 = m - rb ) <= 0 )
        {
            _detail::symm<trans_op>( av, side, uplo, alpha, a.section(extent<S>(m,m)), b.section(extent<S>(m,n)), beta, c.section(extent<S>(m,n)) );
            return;
        }

        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_symm<rb,trans_op>( av, side, uplo, m1, n, alpha, a, b, beta, c );
        gemm( av, transpose::no_trans, transpose::no_trans, m2, n, m1, alpha, a.section(index<S>(m1,0)), b, beta, c.section(index<S>(m1,0)) );
        gemm( av, trans_type, transpose::no_trans, m1, n, m2, alpha, a.section(index<S>(m1,0)), b.section(index<S>(m1,0)), one, c );
        recursive_symm<rb,trans_op>( av, side, uplo, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)), one, c.section(index<S>(m1,0)) );
    }

    // lu case
    else if( side == side::left && uplo == uplo::upper )
    {
        if( ( m1 = m - rb ) <= 0 )
        { 
            _detail::symm<trans_op>( av, side, uplo, alpha, a.section(extent<S>(m,m)), b.section(extent<S>(m,n)), beta, c.section(extent<S>(m,n)) );
            return; 
        }

        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_symm<rb,trans_op>( av, side, uplo, m1, n, alpha, a, b, beta, c );
        gemm( av, transpose::no_trans, transpose::no_trans, m1, n, m2, alpha, a.section(index<S>(0,m1)), b.section(index<S>(m1,0)), one, c );
        gemm( av, trans_type, transpose::no_trans, m2, n, m1, alpha, a.section(index<S>(0,m1)), b, beta, c.section(index<S>(m1,0)) );
        recursive_symm<rb,trans_op>( av, side, uplo, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)), one, c.section(index<S>(m1,0)) );

    }

    // rl case
    else if ( side == side::right && uplo == uplo::lower )
    {
        if( ( n1 = n - rb ) <= 0 )
        { 
            _detail::symm<trans_op>( av, side, uplo, alpha, a.section(extent<S>(n,n)), b.section(extent<S>(m,n)), beta, c.section(extent<S>(m,n)) );
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_symm<rb,trans_op>( av, side, uplo, m, n1, alpha, a, b, beta, c );
        gemm( av, transpose::no_trans, transpose::no_trans, m, n1, n2, alpha, b.section(index<S>(0,n1)), a.section(index<S>(n1,0)), one, c );
        gemm( av, transpose::no_trans, trans_type, m, n2, n1, alpha, b, a.section(index<S>(n1,0)), beta, c.section(index<S>(0,n1)) );
        recursive_symm<rb,trans_op>( av, side, uplo, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)), one, c.section(index<S>(0,n1)) );
    }
    
    // ru case
    else if ( side == side::right && uplo == uplo::upper ) 
    {
        if( ( n1 = n - rb ) <= 0 )
        { 
            _detail::symm<trans_op>( av, side, uplo, alpha, a.section(extent<S>(n,n)), b.section(extent<S>(m,n)), beta, c.section(extent<S>(m,n)) );
            return; 
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_symm<rb,trans_op>( av, side, uplo, m, n1, alpha, a, b, beta, c );
        gemm( av, transpose::no_trans, transpose::no_trans, m, n2, n1, alpha, b, a.section(index<S>(0,n1)), beta, c.section(index<S>(0,n1)) );
        gemm( av, transpose::no_trans, trans_type, m, n1, n2, alpha, b.section(index<S>(0,n1)), a.section(index<S>(0,n1)), one, c );
        recursive_symm<rb,trans_op>( av, side, uplo, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)), one, c.section(index<S>(0,n1)) );
    }
}

template <typename trans_op, typename scalar_type, typename a_type, typename b_type, typename c_type>
void symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, scalar_type alpha, const a_type& a_mat, const b_type& b_mat, scalar_type beta, const c_type& c_mat)
{
    // tuning parameters
    const int tile_size = 16;

    // main routine
    _detail::symm<tile_size,trans_op>(av, side, uplo, alpha, a_mat, b_mat, beta, c_mat);
}


} // namespace _detail

template <typename trans_op, typename scalar_type, typename a_type, typename b_type, typename c_type>
void symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, scalar_type alpha, const a_type& a_mat, const b_type& b_mat, scalar_type beta, const c_type& c_mat)
{
    // extract sizes
    const enum class order S = order::col_major;
    const int m = _detail::rows<S>(c_mat.extent);
    const int n = _detail::columns<S>(c_mat.extent);

    // recursive size
    const int rb = 1024;

    // pass to recursive interface
    _detail::recursive_symm<rb,trans_op>(av, side, uplo, m, n, alpha, a_mat, b_mat, beta, c_mat);
}

template <typename trans_op, typename scalar_type, typename a_type, typename b_type, typename c_type>
void symm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, int m, int n, scalar_type alpha, const a_type& a_mat, const b_type& b_mat, scalar_type beta, const c_type& c_mat)
{
    // build extents
    const order S = order::col_major;
    concurrency::extent<2> a_extent = (side == side::left ? _detail::extent<S>(m,m) : _detail::extent<S>(n,n));
    concurrency::extent<2> bc_extent = _detail::extent<S>(m,n);

    // pass to unsized interface
    symm<trans_op>(av, side, uplo, alpha, a_mat.section(a_extent), b_mat.section(bc_extent), beta, c_mat.section(bc_extent));
}

} // namespace ampblas
