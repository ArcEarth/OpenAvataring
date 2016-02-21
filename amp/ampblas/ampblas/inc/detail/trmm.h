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
 * trmm.h
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_TRMM_H
#define AMPBLAS_TRMM_H

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

template <int tile_size, typename scalar_type, typename a_type, typename b_type, typename c_type>
void trmm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a_mat, const b_type& b_mat, const c_type& c_mat )
{
    const order S = order::col_major;
    const int m = rows<S>(b_mat.extent);
    const int n = columns<S>(b_mat.extent);

    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<b_type>::value, "b_type must be an array_view" ); 
    static_assert( is_array_view<c_type>::value, "c_type must be an array_view" ); 

    // pad() has undesirable functionality - pads even when unnecessary
    // auto e = c_mat.extent.tile<16,16>().pad();

    int tiles_m = (m+tile_size-1)/tile_size;
    int tiles_n = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size*tiles_m, tile_size*tiles_n);

    if (side == side::left)
    {
        if ((uplo == uplo::lower) ^ (transa != transpose::no_trans))
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static scalar_type a[tile_size][tile_size]; // "a" tile
                    tile_static scalar_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_i = idx_b.tile[1];
                    const int tile_i_origin = idx_b.tile_origin[1];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    scalar_type& a_local = (transa == transpose::no_trans ? a[i][j] : a[j][i]);

                    scalar_type out = scalar_type();

                    auto tile_origin = 0;
                    for ( auto tile=0; tile<=tile_i; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_i )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j, tile_origin+i);
                            if ((transa == transpose::no_trans && i >= j) || (transa != transpose::no_trans && i <= j ))
                                a_local = (diag == diag::unit && i == j) ? scalar_type(1) : guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = scalar_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if ( transa == transpose::no_trans )
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,global_i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_i_origin+j,tile_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(global_j, tile_origin+i);
                        b[i][j] = guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,m-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*a[i][k]*b[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }
                    if ( global_i<m && global_j<n )
                    {
                        c_mat[idx_b] = out;
                    }
                }
            );
        }
        else // upper + notrans or lower + trans
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static scalar_type a[tile_size][tile_size]; // "a" tile
                    tile_static scalar_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_i = idx_b.tile[1];
                    const int tile_i_origin = idx_b.tile_origin[1];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    scalar_type& a_local = (transa == transpose::no_trans ? a[i][j] : a[j][i]);

                    scalar_type out = scalar_type();

                    auto tile_origin = tile_i*tile_size;
                    for ( auto tile=tile_i; tile<tiles_m; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_i )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == transpose::no_trans && i <= j ) || ( transa != transpose::no_trans && i >= j ) )
                                a_local = ( diag == diag::unit && i==j ) ? scalar_type(1) : guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = scalar_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if ( transa == transpose::no_trans )
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,global_i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_i_origin+j,tile_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(global_j, tile_origin+i);
                        b[i][j] = guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,m-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*a[i][k]*b[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }
                    if ( global_i<m && global_j<n )
                    {
                        c_mat[idx_b] = out;
                    }
                }
            );
        }
    }
    else // right
    {
        if ((uplo == uplo::lower) ^ (transa != transpose::no_trans))
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static scalar_type a[tile_size][tile_size]; // "a" tile
                    tile_static scalar_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_j = idx_b.tile[0];
                    const int tile_j_origin = idx_b.tile_origin[0];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    scalar_type& a_local = (transa == transpose::no_trans ? a[i][j] : a[j][i]);

                    scalar_type out = scalar_type();

                    auto tile_origin = tile_j*tile_size;
                    for ( auto tile=tile_j; tile<tiles_n; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_j )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == transpose::no_trans && i >= j ) || ( transa != transpose::no_trans && i <= j ) )
                                a_local = ( diag == diag::unit && i==j ) ? scalar_type(1) : guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = scalar_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if ( transa == transpose::no_trans )
                            {
                                auto a_idx = concurrency::index<2>(global_j,tile_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,tile_j_origin+i);
                                a_local = guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(tile_origin+j,global_i);
                        b[i][j] = guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,n-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*b[i][k]*a[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }

                    guarded_write<true>(c_mat, idx_b, out);
                }
            );
        }
        else // upper + notrans or lower + trans
        {
            concurrency::parallel_for_each (
                av,
                e.tile<tile_size,tile_size>(),
                [=] (concurrency::tiled_index<tile_size,tile_size> idx_b) restrict(amp)
                {
                    tile_static scalar_type a[tile_size][tile_size]; // "a" tile
                    tile_static scalar_type b[tile_size][tile_size]; // "b" tile

                    const int i = idx_b.local[1];
                    const int j = idx_b.local[0];
                    const int tile_j = idx_b.tile[0];
                    const int tile_j_origin = idx_b.tile_origin[0];
                    const int global_i = idx_b.global[1];
                    const int global_j = idx_b.global[0];

                    scalar_type& a_local = (transa == transpose::no_trans ? a[i][j] : a[j][i]);

                    scalar_type out = scalar_type();

                    auto tile_origin = 0;
                    for ( auto tile=0; tile<=tile_j; ++tile, tile_origin+=tile_size )
                    {
                        if ( tile == tile_j )
                        {
                            // diagonal tile, treat specially
                            auto a_idx = concurrency::index<2>(tile_origin+j,tile_origin+i);
                            if ( ( transa == transpose::no_trans && i <= j ) || ( transa != transpose::no_trans && i >= j ) )
                                a_local = ( diag == diag::unit && i==j ) ? scalar_type(1) : _detail::guarded_read<true>(a_mat,a_idx);
                            else
                                a_local = scalar_type();
                        }
                        else
                        {
                            // off diagonal, load
                            if (transa == transpose::no_trans)
                            {
                                auto a_idx = concurrency::index<2>(global_j,tile_origin+i);
                                a_local = _detail::guarded_read<true>(a_mat,a_idx);
                            }
                            else
                            {
                                auto a_idx = concurrency::index<2>(tile_origin+j,tile_j_origin+i);
                                a_local = _detail::guarded_read<true>(a_mat,a_idx);
                            }
                        }
                        auto b_idx = concurrency::index<2>(tile_origin+j,global_i);
                        b[i][j] = _detail::guarded_read<true>(b_mat,b_idx);

                        idx_b.barrier.wait_with_tile_static_memory_fence();

                        int end = _detail::min(tile_size,n-tile_origin);
                        for ( int k=0; k<end; ++k )
                            out += alpha*b[i][k]*a[k][j];

                        idx_b.barrier.wait_with_tile_static_memory_fence();
                    }

                    guarded_write<true>(c_mat, idx_b, out);
                }
            );
        }
    }
}

// recursive gemm-based implementation
template <int rb, typename scalar_type, typename a_type, typename b_type, typename c_type>
void recursive_trmm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b, const c_type& c)
{
    // TODO: this requires a true B=A*B TRMM kernel

    const order S = order::col_major;

    int m1, m2, n1, n2;
    scalar_type one = one<scalar_type>::value;

    // lun
    if ( side == side::left && uplo == uplo::upper && transa == transpose::no_trans )
    {
        if ( ( m1 = m - rb ) <= 0 )
        {
            _detail::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m1, n, alpha, a, b );
        gemm( transpose::no_trans, transpose::no_trans, m1, n, m2, alpha, a.section(index<S>(0,m1)), b.section(index<S>(m1,0)), one, b );
        recursive_trmm<rb>( side, uplo, transa, diag, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
    }  

    // lut / luc
    else if ( side == side::left && uplo == uplo::upper && transa != transpose::no_trans )
    {
        if( ( m1 = m - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
        gemm( transa, transpose::no_trans, m2, n, m1, alpha, a.section(index<S>(0,m1)), b, one, b.section(index<S>(m1,0)) );
        recursive_trmm<rb>( side, uplo, transa, diag, m1, n, alpha, a, b );
    }  

    // lln
    else if ( side == side::left && uplo == uplo::lower && transa == transpose::no_trans )
    {
        if( ( m1 = m - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
        gemm( transpose::no_trans, transpose::no_trans, m2, n, m1, alpha, a+m1, b, one, b.section(index<S>(m1,0)) );
        recursive_trmm<rb>( side, uplo, transa, diag, m1, n, alpha, a, b );
    }    

    // llt / llc
    else if ( side == side::left && uplo == uplo::lower && transa != transpose::no_trans )
    {
        if( ( m1 = m - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m1, n, alpha, a, b );
        gemm( transa, transpose::no_trans, m1, n, m2, alpha, a.section(index<S>(m1,0)), b.section(index<S>(m1,0)), one, b );
        recursive_trmm<rb>( side, uplo, transa, diag, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
    }   

    // run
    else if ( side == side::right && uplo == uplo::upper && transa == transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
        gemm( transpose::no_trans, transpose::no_trans, m, n2, n1, alpha, b, a.section(index<S>(0,n1)), one, b.section(index<S>(0,n1)) );
        recursive_trmm<rb>( side, uplo, transa, diag, m, n1, alpha, a, b );
    }

    // rut / ruc
    else if ( side == side::right && uplo == uplo::upper && transa != transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m, n1, alpha, a, b );
        gemm( transpose::no_trans, transa, m, n1, n2, alpha, b.section(index<S>(0,n1)), a.section(index<S>(0,n1)), one, b );
        recursive_trmm<rb>( side, uplo, transa, diag, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
    }

    // rln
    else if ( side == side::right && uplo == uplo::lower && transa == transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m, n1, alpha, a, b );
        gemm( transpose::no_trans, transpose::no_trans, m, n1, n2, alpha, b.section(index<S>(0,n1)), a.section(index<S>(n1,0)), one, b );
        recursive_trmm<rb>( side, uplo, transa, diag, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
    }

    // rlt / rlc
    else if ( side == side::right && uplo == uplo::lower && transa != transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        {
            cublas::trmm( side, uplo, transa, diag, m, n, alpha, a, b ); 
            return;
        }

        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trmm<rb>( side, uplo, transa, diag, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
        gemm( transpose::no_trans, transa, m, n2, n1, alpha, b, a.section(index<S>(n1,0)), one, b.section(index<S>(0,n1)) );
        recursive_trmm<rb>( side, uplo, transa, diag, m, n1, alpha, a, b );
    }
}

// tuning function
template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void trmm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b, const c_type& c)
{
    // tuning parameters
    const int tile_size = 16;

    // call implementation
    _detail::trmm<tile_size>(av, side, uplo, transa, diag, alpha, a, b, c);
}

} // namespace _detail

template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void trmm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b, const c_type& c)
{
    // pass to tuning funcitons
    _detail::trmm(av, side, uplo, transa, diag, alpha, a, b, c);
}

template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void trmm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, int m, int n, scalar_type alpha, const a_type& a, const b_type& b, const c_type& c)
{
    const order S = order::col_major;

    const int k = (side == side::left ? m : n);

    concurrency::extent<2> a_extent = _detail::extent<S>(k, k);
    concurrency::extent<2> b_extent = _detail::extent<S>(m, n);
    concurrency::extent<2> c_extent = _detail::extent<S>(m, n);
    
    // pass sections to the unsized interface
    trmm(av, side, uplo, transa, diag, alpha, a.section(a_extent), b.section(b_extent), c.section(c_extent));
}

} // namespace ampblas

#endif // AMPBLAS_TRMM_H
