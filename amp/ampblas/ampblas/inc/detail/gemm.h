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
 * gemm.h
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_GEMM_H
#define AMPBLAS_GEMM_H

#include "ampblas_config.h"
#include "ampblas_utility.h"

#include "tuning/gemm.h"

namespace ampblas {
namespace _detail {

//
// Execution Pipeline
//

// Stage 1: Refactor as row major implementation (row major can skip to stage 2)
template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm(const concurrency::accelerator_view& av, enum class transpose transa, enum class transpose transb, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    // transa <==> transb
    // a <==> b
    gemm_stage_2(av, transb, transa, alpha, b, a, beta, c);
}

// Stage 2: Hardcoded architecture as template parameter
template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm_stage_2(const concurrency::accelerator_view& av, enum class transpose transa, enum class transpose transb, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    // obtain architecture based off information in the accelerator_view
    std::wstring desc = av.accelerator.get_description();
    const enum class architecture arch = get_architecture(desc);
    
    if (arch == architecture::amd)
    {
        gemm_stage_3<architecture::amd>(av, transa, transb, alpha, a, b, beta, c);
    }
    else if (arch == architecture::nvidia)
    {
        gemm_stage_3<architecture::nvidia>(av, transa, transb, alpha, a, b, beta, c);
    }
    else
    {
        gemm_stage_3<architecture::unknown>(av, transa, transb, alpha, a, b, beta, c);
    }
}

// Stage 3: Hardcoded transpose operations as template parameters
template <enum class architecture arch, typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm_stage_3(const concurrency::accelerator_view& av, enum class transpose transa, enum class transpose transb, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    if (transa == transpose::no_trans)
    {
        if (transb == transpose::no_trans)
        {
            // NN
            gemm_stage_4<arch, transpose::no_trans, transpose::no_trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::trans)
        {
            // NT
            gemm_stage_4<arch, transpose::no_trans, transpose::trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::conj_trans)
        {
            // NC
            gemm_stage_4<arch, transpose::no_trans, transpose::conj_trans>(av, alpha, a, b, beta, c);
        }
    }
    else if (transa == transpose::trans)
    {
        if (transb == transpose::no_trans)
        {
            // TN
            gemm_stage_4<arch, transpose::trans, transpose::no_trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::trans)
        {
            // TT
            gemm_stage_4<arch, transpose::trans, transpose::trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::conj_trans)
        {
            // TC
            gemm_stage_4<arch, transpose::trans, transpose::conj_trans>(av, alpha, a, b, beta, c);
        }
    }
    else if (transa == transpose::conj_trans)
    {
        if (transb == transpose::no_trans)
        {
            // CN
            gemm_stage_4<arch, transpose::conj_trans, transpose::no_trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::trans)
        {
            // CT
            gemm_stage_4<arch, transpose::conj_trans, transpose::trans>(av, alpha, a, b, beta, c);
        }
        else if (transb == transpose::conj_trans)
        {
            // CC
            gemm_stage_4<arch, transpose::conj_trans, transpose::conj_trans>(av, alpha, a, b, beta, c);
        }
    }
}

// Stage 4: find tuning parameters, check if we need an IO guard, and finally pass to the kernel!
template <enum class architecture arch, enum class transpose transa, enum class transpose transb, typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm_stage_4(const concurrency::accelerator_view& av, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)  
{ 
    // alias to all important tuning parameters
    typedef gemm_tuning_parameters<arch, scalar_type, transa, transb> tp;

    // row major
    const int m = c.extent[0];  
    const int n = c.extent[1];
    const int k = (transa != transpose::no_trans ? a.extent[0] : a.extent[1]); 

    if (tp::m_block % m || tp::n_block % n || tp::k_block % k) 
    {
        // one or more dimensions doesn't align with work block size, must use IO guards
        const bool guarded = true;
        gemm_kernel<guarded, transa, transb, tp::m_block, tp::n_block, tp::k_block, tp::m_c_tile, tp::n_c_tile, tp::m_a_tile, tp::n_a_tile, tp::m_b_tile, tp::n_b_tile, tp::use_padding>(av, alpha, a, b, beta, c);
    }
    else
    {
        // all dimensions align; safe to skip bounds checks
        const bool guarded = false;
        gemm_kernel<guarded, transa, transb, tp::m_block, tp::n_block, tp::k_block, tp::m_c_tile, tp::n_c_tile, tp::m_a_tile, tp::n_a_tile, tp::m_b_tile, tp::n_b_tile, tp::use_padding>(av, alpha, a, b, beta, c);
    }
}

// Stage 5: Highly parameterized GEMM
template <bool guarded, enum class transpose transa, enum class transpose transb, int m_block, int n_block, int k_block, int m_c_tile, int n_c_tile, int m_a_tile, int n_a_tile, int m_b_tile, int n_b_tile, int use_padding, typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm_kernel(const concurrency::accelerator_view& av, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    // only possibly on array_views
    static_assert( is_array_view<a_type>::value, "a_type must be an array_view" ); 
    static_assert( is_array_view<b_type>::value, "b_type must be an array_view" ); 
    static_assert( is_array_view<c_type>::value, "c_type must be an array_view" ); 

    // static checks for block usage
    static_assert( m_block % (transa == transpose::no_trans ? m_a_tile : n_a_tile) == 0, "static tuning error: a tile must evenly divide into [m x k] work block");
    static_assert( k_block % (transa == transpose::no_trans ? n_a_tile : m_a_tile) == 0, "static tuning error: a tile must evenly divide into [m x k] work block");
    static_assert( k_block % (transb == transpose::no_trans ? m_b_tile : n_b_tile) == 0, "static tuning error: b tile must evenly divide into [k x n] work block");
    static_assert( n_block % (transb == transpose::no_trans ? n_b_tile : m_b_tile) == 0, "static tuning error: b tile must evenly divide into [k x n] work block");
    static_assert( m_block % m_c_tile == 0, "static tuning error: c tile must evenly divide into [m x n] work block");
    static_assert( n_block % n_c_tile == 0, "static tuning error: c tile must evenly divide into [m x n] work block");

    // derived tuning parameters
    static const int m_thread = m_block / m_c_tile;
    static const int n_thread = n_block / n_c_tile;

    // row major!
    const int M = c.extent[0];  
    const int N = c.extent[1];
    const int K = (transa != transpose::no_trans ? a.extent[0] : a.extent[1]); 

    // build extent
    const int m_extent = ((M + (m_block-1)) / m_block) * m_c_tile;
    const int n_extent = ((N + (n_block-1)) / n_block) * n_c_tile;
    concurrency::extent<2> extent(m_extent, n_extent);

    concurrency::parallel_for_each(
        av,
        extent.tile<m_c_tile, n_c_tile>(), 
        [=] (concurrency::tiled_index<m_c_tile, n_c_tile> tid) restrict(amp)
    {
        // shared memory padding to (potentially) reduce bank conflicts 
        const int a_padding = (use_padding && transa != transpose::no_trans ? 1 : 0);
        const int b_padding = (use_padding && transb != transpose::no_trans ? 1 : 0);

        // shared tile memory
		tile_static scalar_type a_tile[m_block][k_block + a_padding];
		tile_static scalar_type b_tile[k_block][n_block + b_padding];

        // global tile offset indexing
		const int i = tid.tile[0] * m_block;    
		const int j = tid.tile[1] * n_block;

        // local c indexing [m_c_tile x n_c_tile]
		const int i_c_idx = tid.local[0];
		const int j_c_idx = tid.local[1]; 

        // 1D index 
        // TODO: is there a built in way to 
        const int idx = i_c_idx * n_c_tile + j_c_idx;    

        // local a index [m_a_tile x n_a_tile]
        const int i_a_idx = idx / n_a_tile;
        const int j_a_idx = idx % n_a_tile;

        // local b index [m_b_tile x n_b_tile]
        const int i_b_idx = idx / n_b_tile;
        const int j_b_idx = idx % n_b_tile; 

        // local computation registers
        scalar_type c_reg[m_thread][n_thread]; 
        scalar_type a_reg[m_thread];
        scalar_type b_reg[n_thread];

        // zero out the sumation registers
        for (int m = 0; m < m_thread; m++)
            for (int n = 0; n < n_thread; n++)
                c_reg[m][n] = scalar_type();

        // outer k-loop
        for (int ko = 0; ko < K; ko += k_block)
        {
            // read a [m_block x k_block] into shared memory using tile [m_a_tile x n_a_tile]
            if (transa == transpose::no_trans)
            {
                for (int m = 0; m < m_block; m += m_a_tile)
                    for (int n = 0; n < k_block; n += n_a_tile)
                        a_tile[m+i_a_idx][n+j_a_idx] = guarded_read<guarded>(a, concurrency::index<2>(i+m+i_a_idx, ko+n+j_a_idx)); 
            }
            else
            {
                for (int n = 0; n < m_block; n += n_a_tile)
                    for (int m = 0; m < k_block; m += m_a_tile)
                        a_tile[n+j_a_idx][m+i_a_idx] = guarded_read<guarded>(a, concurrency::index<2>(ko+m+i_a_idx, i+n+j_a_idx));
            }

            // read b [k_block x n_block] into shared memory using tile [m_b_tile x n_b_tile]
            if (transb == transpose::no_trans)
            {
			    for (int m = 0; m < k_block; m += m_b_tile)
				    for (int n = 0; n < n_block; n += n_b_tile)
					    b_tile[m+i_b_idx][n+j_b_idx] = guarded_read<guarded>(b, concurrency::index<2>(ko+m+i_b_idx, j+n+j_b_idx));
            }
            else
            {
                for (int n = 0; n < k_block; n += n_b_tile)
                    for (int m = 0; m < n_block; m += m_b_tile)
                        b_tile[n+j_b_idx][m+i_b_idx] = guarded_read<guarded>(b, concurrency::index<2>(j+m+i_b_idx, ko+n+j_b_idx));
            }

            // wait for tiled static memory to fill
            tid.barrier.wait_with_tile_static_memory_fence();

            // inner k-loop
            for (int ki = 0; ki < k_block; ki++)
            {
                // load a registers
                for (int m = 0; m < m_thread; m++)
                    a_reg[m] = a_tile[m*m_c_tile+i_c_idx][ki];
                
                // load b registers
                for (int n = 0; n < n_thread; n++)
                    b_reg[n] = b_tile[ki][n*n_c_tile+j_c_idx];
                
                // accumulate into c registers
                for (int m = 0; m < m_thread; m++)
                    for (int n = 0; n < n_thread; n++)
                        c_reg[m][n] += (transa == transpose::conj_trans ? conjugate::op(a_reg[m]) : a_reg[m]) * (transb == transpose::conj_trans ? conjugate::op(b_reg[n]) : b_reg[n]);
            }

            // wait for tiled static memory to be consumed
            tid.barrier.wait_with_tile_static_memory_fence();
        
        } // outer k-loop

        // write registers to c
		for (int m = 0; m < m_thread; m++)
		{
			const int m_out = i + (m*m_c_tile+i_c_idx); 
			for (int n = 0; n < n_thread; n++)
			{
				const int n_out = j + (n*n_c_tile+j_c_idx);

                scalar_type c_temp = guarded_read<guarded>(c, concurrency::index<2>(m_out, n_out));
				c_temp = alpha*c_reg[m][n] + beta*c_temp;
                guarded_write<guarded>(c, concurrency::index<2>(m_out, n_out), c_temp);
			}
		}
    });
}

} // namespace _detail

// Sections of the A, B, and C matrices are specified in the interface via the m, n, k sizes
template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm(const concurrency::accelerator_view& av, enum class transpose transa, enum class transpose transb, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    // pass to tuning pipeline
    _detail::gemm(av, transa, transb, alpha, a, b, beta, c);
}

// Sections of the A, B, and C matrices are specified in the interface via the m, n, k sizes
template <typename scalar_type, typename a_type, typename b_type, typename c_type>
void gemm(const concurrency::accelerator_view& av, enum class transpose transa, enum class transpose transb, int m, int n, int k, scalar_type alpha, const a_type& a, const b_type& b, scalar_type beta, const c_type& c)
{
    // only column-major storage supported
    const order S = order::col_major;

    // a = m x k
    int a_row = m;
    int a_col = k;
    if (transa != transpose::no_trans)
        std::swap(a_row, a_col);
    concurrency::extent<2> a_extent = _detail::extent<S>(a_row, a_col);

    // b = k x n
    int b_row = k;
    int b_col = n;
    if (transb != transpose::no_trans)
         std::swap(b_row, b_col);
    concurrency::extent<2> b_extent = _detail::extent<S>(b_row, b_col);

    // c = m x n
    concurrency::extent<2> c_extent = _detail::extent<S>(m, n);

    // pass sections to the unsized interface
    gemm(av, transa, transb, alpha, a.section(a_extent), b.section(b_extent), beta, c.section(c_extent)); 
}

} // namespace ampblas

#endif // AMPBLAS_GEMM_H
