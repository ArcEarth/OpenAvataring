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
 * trsm.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

#include "gemm.h"

namespace ampblas {
namespace _detail {

template <typename scalar_type>
struct trsm_tile_size { static const int value = 16; };

template <>
struct trsm_tile_size<complex<double>> { static const int value = 16; };

template <int tile_size, bool guarded, typename scalar_type, typename a_type, typename b_type>
void trsm_ll(const concurrency::accelerator_view& av, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b) 
{
    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<b_type>::value, "b_type must be an array_view" ); 

    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    concurrency::parallel_for_each ( 
        av,
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
    {
        // shared memory buffers
        tile_static scalar_type a_tile[tile_size][tile_size];
        tile_static scalar_type b_tile[tile_size][tile_size];

        // local indexes
        const int col = tid.local[0];
        const int row = tid.local[1];

        // per-thread common alias
        // transpose read pattern still allows for coalesced global memory access
        scalar_type& a_local = (transa == transpose::no_trans ? a_tile[row][col] : a_tile[col][row]);
        scalar_type& b_local = b_tile[row][col];

        // global j index
        const int j = tid.tile_origin[0];

        // loop down by tiles
        for (int i=0; i<m; i+=tile_size)
        {
            // read tile at A(i,i) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(i+row, i+col));

            // apply conjugation
            if (transa == transpose::conj_trans)
                a_local = conjugate::op(a_local);

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait();

            // solve X(i,j) = B(i,j) \ A(i,i)
            if (col == 0)
            {
                int jj = row;

                // loop down shared block
                for (int ii=0; ii<tile_size; ii++)
                {
                    // elimation scalar
                    scalar_type temp = b_tile[jj][ii];
                    if (diag == diag::non_unit)
                        temp /= a_tile[ii][ii];

                    // apply
                    for (unsigned int kk=ii+1; kk<tile_size; kk++)
                        b_tile[jj][kk] -= temp * a_tile[ii][kk];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait();

            // apply B(k,j) -= B(i,j) * A(k,i) 
            for (int k=i+tile_size; k<m; k+=tile_size)
            {   
                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == transpose::no_trans ? concurrency::index<2>(i+row, k+col) : concurrency::index<2>(k+row, i+col));

                // apply conjugation
                if (transa == transpose::conj_trans)
                    a_local = conjugate::op(a_local);

                tid.barrier.wait();

                // accumulate
                scalar_type sum = scalar_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += a_tile[l][col] * b_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(j+row, k+col), _detail::subtract<scalar_type>(sum));

                // wait for a to finish being read
                tid.barrier.wait();
            }

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);
        }
    });
}

template <int tile_size, bool guarded, typename scalar_type, typename a_type, typename b_type>
void trsm_lu(const concurrency::accelerator_view& av, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b) 
{
    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<b_type>::value, "b_type must be an array_view" ); 

    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (n+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    // compiler work around
    const int dummy = 1;

    concurrency::parallel_for_each ( 
        av,
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
    {
        // shared memory buffers
        tile_static scalar_type a_tile[tile_size][tile_size];
        tile_static scalar_type b_tile[tile_size][tile_size];

        // local indexes
        const int col = tid.local[0];
        const int row = tid.local[1];

        // per-thread common alias
        // transpose read pattern still allows for coalesced global memory access
        scalar_type& a_local = (transa == transpose::no_trans ? a_tile[row][col] : a_tile[col][row]);
        scalar_type& b_local = b_tile[row][col];

        // global j index
        const int j = tid.tile_origin[0];

        // loop up by tiles
        for (int i_ = (m-1) & (-tile_size); i_>=0; i_-=tile_size)
        {
            // compiler work around
            int i = dummy ? i_ : 0;            

            // read tile at A(i,i) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(i+row, i+col));

            // apply conjugation
            if (transa == transpose::conj_trans)
                a_local = conjugate::op(a_local);

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait_with_tile_static_memory_fence();

            // solve X(i,j) = B(i,j) \ A(i,i)
            if (col == 0)
            {
                int jj = row;

                // loop down shared block
                for (int ii=_detail::min(tile_size-1,m-1-i); ii>=0; ii--)
                {
                    // elimation scalar
                    scalar_type temp = b_tile[jj][ii];

                    if (diag == diag::non_unit)
                        temp /= (a_tile[ii][ii] == scalar_type() ? scalar_type(1) : a_tile[ii][ii]);

                    // apply
                    for (int kk=0; kk<ii; kk++)
                        b_tile[jj][kk] -= temp * a_tile[ii][kk];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait_with_tile_static_memory_fence();

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);

            // apply B(k,j) -= B(i,j) * A(k,i) 
            for (int k_=0; k_<i; k_+=tile_size)
            {   
                // compiler workaround
                int k = dummy ? k_ : 0;

                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == transpose::no_trans ? concurrency::index<2>(i+row, k+col) : concurrency::index<2>(k+row, i+col));

                // apply conjugation
                if (transa == transpose::conj_trans)
                    a_local = conjugate::op(a_local);

                tid.barrier.wait_with_tile_static_memory_fence();

                // accumulate
                scalar_type sum = scalar_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += a_tile[l][col] * b_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(j+row, k+col), _detail::subtract<scalar_type>(sum));
                tid.barrier.wait_with_tile_static_memory_fence();
            }
        }
    });
}

template <int tile_size, bool guarded, typename scalar_type, typename a_type, typename b_type>
void trsm_rl(const concurrency::accelerator_view& av, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b)
{
    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<b_type>::value, "b_type must be an array_view" ); 

    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (m+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    // compiler work around
    const int dummy = 1;

    concurrency::parallel_for_each ( 
        av,
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
    {
        // shared memory buffers
        tile_static scalar_type a_tile[tile_size][tile_size];
        tile_static scalar_type b_tile[tile_size][tile_size];

        // local indexes
        const int col = tid.local[0];
        const int row = tid.local[1];

        // per-thread common alias
        // transpose read pattern still allows for coalesced global memory access
        scalar_type& a_local = (transa == transpose::no_trans ? a_tile[row][col] : a_tile[col][row]);
        scalar_type& b_local = b_tile[row][col];
        
        // global i index
        const int i = tid.tile_origin[0];

        // loop right to left across tiles
        for (int j_=(n-1) & (-tile_size); j_>=0; j_-=tile_size)
        {
            // compiler work around
            int j = dummy ? j_ : 0;

            // read tile at A(j,j) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(j+row, j+col));

            // apply conjugation
            if (transa == transpose::conj_trans)
                a_local = conjugate::op(a_local);

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait_with_tile_static_memory_fence();

            // solve A(j,j) * X(i,j) = B(i,j)
            if (col == 0)
            {
                int ii = row;

                // loop down shared block
                for (int jj=_detail::min(tile_size-1,n-1-j); jj>=0; jj--)
                {
                    // elimation scalar
                    scalar_type temp = b_tile[jj][ii];

                    if (diag == diag::non_unit)
                        temp /= (a_tile[jj][jj] == scalar_type() ? scalar_type(1) : a_tile[jj][jj]);

                    // apply
                    for (int kk=0; kk<jj; kk++)
                        b_tile[kk][ii] -= temp * a_tile[kk][jj];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait_with_tile_static_memory_fence();

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);
            tid.barrier.wait_with_tile_static_memory_fence();

            // apply B(i,k) -= A(j,k) * B(i,j)
            for (int k_=0; k_<j; k_+=tile_size)
            {   
                // compiler work around
                int k = dummy ? k_ : 0;

                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == transpose::no_trans ? concurrency::index<2>(k+row, j+col) : concurrency::index<2>(j+row, k+col));

                // apply conjugation
                if (transa == transpose::conj_trans)
                    a_local = conjugate::op(a_local);

                tid.barrier.wait();

                // accumulate
                scalar_type sum = scalar_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += b_tile[l][col] * a_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(k+row, i+col), _detail::subtract<scalar_type>(sum));

                // wait for a to finish being read
                tid.barrier.wait_with_tile_static_memory_fence();
            }
        }

    });
}

template <int tile_size, bool guarded, typename scalar_type, typename a_type, typename b_type>
void trsm_ru(const concurrency::accelerator_view& av, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b) 
{
    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<b_type>::value, "b_type must be an array_view" ); 

    // runtime sizes
    int m = b.extent[1];
    int n = b.extent[0];
    int tiles = (m+tile_size-1)/tile_size;

    // configuration
    auto e = make_extent(tile_size, tile_size*tiles);

    concurrency::parallel_for_each( 
        av,
        e.tile<tile_size,tile_size>(),
        [=] (concurrency::tiled_index<tile_size,tile_size> tid) restrict(amp)
    {
        // shared memory buffers
        tile_static scalar_type a_tile[tile_size][tile_size];
        tile_static scalar_type b_tile[tile_size][tile_size];

        // local indexes
        const int col = tid.local[0];
        const int row = tid.local[1];

        // per-thread common alias
        // transpose read pattern still allows for coalesced global memory access
        scalar_type& a_local = (transa == transpose::no_trans ? a_tile[row][col] : a_tile[col][row]);
        scalar_type& b_local = b_tile[row][col];

        // global i index
        const int i = tid.tile_origin[0];

        // loop right to left across tiles
        for (int j=0; j<n; j+=tile_size)
        {
            // read tile at A(j,j) into local A
            a_local = _detail::guarded_read<guarded>(a, concurrency::index<2>(j+row, j+col));

            // apply conjugation
            if (transa == transpose::conj_trans)
                a_local = conjugate::op(a_local);

            // read tile at B(i,j) into local B
            b_local = _detail::guarded_read<guarded>(b, concurrency::index<2>(j+row, i+col));
            tid.barrier.wait_with_tile_static_memory_fence();

            // solve A(j,j) * X(i,j) = B(i,j)
            if (col == 0)
            {
                int ii = row;

                // loop down shared block
                for (int jj=0; jj<_detail::min(tile_size,n-j);jj++)
                {
                    // elimation scalar
                    scalar_type temp = b_tile[jj][ii];

                    if (diag == diag::non_unit)
                        temp /= (a_tile[jj][jj] == scalar_type() ? scalar_type(1) : a_tile[jj][jj]);

                    // apply
                    for (int kk=jj; kk<tile_size; kk++)
                        b_tile[kk][ii] -= temp * a_tile[kk][jj];

                    b_tile[jj][ii] = temp;
                }
            }

            // wait for local solve
            tid.barrier.wait_with_tile_static_memory_fence();

            // write B(i,j)
            _detail::guarded_write<guarded>(b, concurrency::index<2>(j+row, i+col), alpha*b_local);
            tid.barrier.wait_with_tile_static_memory_fence();

            // apply B(i,k) -= A(j,k) * B(i,j) 
            for (int k=j+tile_size; k<n; k+=tile_size)
            {   
                // read tile at A(k,i) into local A
                a_local = _detail::guarded_read<guarded>(a, transa == transpose::no_trans ? concurrency::index<2>(k+row, j+col) : concurrency::index<2>(j+row, k+col));

                // apply conjugation
                if (transa == transpose::conj_trans)
                    a_local = conjugate::op(a_local);

                tid.barrier.wait_with_tile_static_memory_fence();

                // accumulate
                scalar_type sum = scalar_type();

                // TODO: explictly unrollable?
                for (int l=0; l<tile_size; l++)
                    sum += b_tile[l][col] * a_tile[row][l];

                // update
                _detail::guarded_update<guarded>(b, concurrency::index<2>(k+row, i+col), _detail::subtract<scalar_type>(sum));

                // wait for a to finish being read
                tid.barrier.wait_with_tile_static_memory_fence();
            }
        }
    });
}

// recursive gemm-based implementation
template <int rb, typename scalar_type, typename a_type, typename b_type>
void recursive_trsm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, int m, int n, scalar_type alpha, const a_type& a, const b_type& b) 
{
    // only column major support for now
    const enum class order S = order::col_major;
   
    int m1, m2, n1, n2;

    scalar_type one = scalar_type(1);
    scalar_type negone = scalar_type(-1);

    // RUT / RUC
    if (side == side::right && uplo == uplo::upper && transa != transpose::no_trans)
    {
        if( ( n1 = n - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(n,n)), b.section(extent<S>(m,n)) );
            return; 
        }
        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
        gemm( av, transpose::no_trans, transa, m, n1, n2, negone, b.section(index<S>(0,n1)), a.section(index<S>(0,n1)), alpha, b.section(index<S>(0,0)) );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n1, one, a, b );
    }

    // RUN
    else if ( side == side::right && uplo == uplo::upper && transa == transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(n,n)), b.section(extent<S>(m,n)) );
            return; 
        }
        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n1, alpha, a, b );
        gemm( av, transpose::no_trans, transpose::no_trans, m, n2, n1, negone, b.section(index<S>(0,0)), a.section(index<S>(0,n1)), alpha, b.section(index<S>(0,n1)) );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n2, one, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
    }

    // RLT / RLC
    else if ( side == side::right && uplo == uplo::lower && transa != transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(n,n)), b.section(extent<S>(m,n)) );
            return; 
        }
        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n1, alpha, a, b );
        gemm( av, transpose::no_trans, transa, m, n2, n1, negone, b.section(index<S>(0,0)), a.section(index<S>(n1,0)), alpha, b.section(index<S>(0,n1)) );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n2, one, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
    }


    // RLN
    else if ( side == side::right && uplo == uplo::lower && transa == transpose::no_trans )
    {
        if( ( n1 = n - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(n,n)), b.section(extent<S>(m,n)) );
            return; 
        }
        n2 = n - ( n1 = rb + ( n1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n2, alpha, a.section(index<S>(n1,n1)), b.section(index<S>(0,n1)) );
        gemm( av, transpose::no_trans, transpose::no_trans, m, n1, n2, negone, b.section(index<S>(0,n1)), a.section(index<S>(n1,0)), alpha,  b.section(index<S>(0,0)) );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m, n1, one, a, b );
    }
    
    // LUT / LUC
    else if ( side == side::left && uplo == uplo::upper && transa != transpose::no_trans )
    {
        if( ( m1 = m - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(m,m)), b.section(extent<S>(m,n)) );
            return; 
        }
        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m1, n, alpha, a, b );
        gemm( av, transa, transpose::no_trans, m2, n, m1, negone, a.section(index<S>(0,m1)), b, alpha, b.section(index<S>(m1,0)) );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m2, n, one, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
    }

    // LUN
    else if (side == side::left && uplo == uplo::upper && transa == transpose::no_trans)
    {
        if( ( m1 = m - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(m,m)), b.section(extent<S>(m,n)) );
            return; 
        }
        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
        gemm( av, transpose::no_trans, transpose::no_trans, m1, n, m2, negone, a.section(index<S>(0,m1)), b.section(index<S>(m1,0)), alpha, b );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m1, n, one, a, b );
    }

    // LLT / LLC
    else if ( side == side::left && uplo == uplo::lower && transa != transpose::no_trans )
    {
        if( ( m1 = m - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(m,m)), b.section(extent<S>(m,n)) );
            return; 
        }
        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m2, n, alpha, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
        gemm( av, transa, transpose::no_trans, m1, n, m2, negone, a.section(index<S>(m1,0)), b.section(index<S>(m1,0)), alpha, b );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m1, n, one, a, b );
    }

    // LLN
    else if ( side == side::left && uplo == uplo::lower && transa == transpose::no_trans )
    {
        if( ( m1 = m - rb ) <= 0 )
        { 
            _detail::trsm(av, side, uplo, transa, diag, alpha, a.section(extent<S>(m,m)), b.section(extent<S>(m,n)) );
            return; 
        }
        m2 = m - ( m1 = rb + ( m1 / ( rb << 1 ) ) * rb );

        recursive_trsm<rb>( av, side, uplo, transa, diag, m1, n, alpha, a, b );
        gemm( av, transpose::no_trans, transpose::no_trans, m2, n, m1, negone, a.section(index<S>(m1,0)), b, alpha, b.section(index<S>(m1,0)) );
        recursive_trsm<rb>( av, side, uplo, transa, diag, m2, n, one, a.section(index<S>(m1,m1)), b.section(index<S>(m1,0)) );
    }
}

// tuning dispatch function
template <typename scalar_type, typename a_type, typename b_type>
void trsm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b) 
{
    // tuning parameters
    const int tile_size = trsm_tile_size<scalar_type>::value;
    const bool guarded = true;

    // select proper kernel based on options
    if (side == side::left)
    {
        if ((uplo == uplo::lower) ^ (transa != transpose::no_trans))
        {
            // lower + no trans <==> upper + trans 
            _detail::trsm_ll<tile_size,guarded>(av, transa, diag, alpha, a, b);
        }
        else
        {
            // upper + no trans <==> lower + trans 
            _detail::trsm_lu<tile_size,guarded>(av, transa, diag, alpha, a, b);
        }
    }
    else if (side == side::right)
    {
        if ((uplo == uplo::lower) ^ (transa != transpose::no_trans))
        {
            // lower + no trans <==> upper + trans 
            _detail::trsm_rl<tile_size,guarded>(av, transa, diag, alpha, a, b);
        }
        else
        {
            // upper + no trans <==> lower + trans
            _detail::trsm_ru<tile_size,guarded>(av, transa, diag, alpha, a, b);
        }
    }
}

} // namespace _detail

// use full matrices from A and B
template <typename scalar_type, typename a_type, typename b_type>
void trsm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, scalar_type alpha, const a_type& a, const b_type& b)
{
    // only column major supported for now
    const order S = order::col_major;

    // recursive cross over point
    const int rb = 1024;

    // forward to recursive function
    const int m = _detail::rows<S>(b.extent);
    const int n = _detail::columns<S>(b.extent);
    _detail::recursive_trsm<rb>(av, side, uplo, transa, diag, m, n, alpha, a, b);
}

// use sections of A and B specified by m and n
template <typename scalar_type, typename a_type, typename b_type>
void trsm(const concurrency::accelerator_view& av, enum class side side, enum class uplo uplo, enum class transpose transa, enum class diag diag, int m, int n, scalar_type alpha, const a_type& a, const b_type& b)
{
    // only column major supported for now
    const order S = order::col_major;

    // size check of A
    concurrency::extent<2> a_extent = _detail::extent<S>(k, k);
    concurrency::extent<2> b_extent = _detail::extent<S>(m, n);
    
    // forward to unsized function
    trsm(av, side, uplo, transa, alpha, a.section(a_extent), b.section(b_extent));
}

} // namespace ampblas
