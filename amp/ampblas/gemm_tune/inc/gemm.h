#ifndef AMPBLAS_GEMM_PROFILE_GEMM_H
#define AMPBLAS_GEMM_PROFILE_GEMM_H

#include "tune.h"

TUNE_NAMESPACE_BEGIN

//
// NOTE: All indexing in this implementation is column major
//

template <typename value_type>
inline value_type conj(const value_type& x) restrict(amp,cpu)
{
    // noop
    return x;
}

template <bool enabled, typename value_type>
inline value_type guarded_read(const concurrency::array_view<value_type,2>& a, const concurrency::index<2>& idx) restrict(cpu,amp)
{
    if (!enabled || a.extent.contains(idx))
        return a[idx];
    else
        return value_type();
}

template <bool enabled, typename value_type>
inline void guarded_write(const concurrency::array_view<value_type,2>& a, const concurrency::index<2>& idx, const value_type& val) restrict(cpu,amp)
{
    if (!enabled || a.extent.contains(idx))
        a[idx] = val;
}

template <typename value_type, bool guarded, enum class transpose transa, enum class transpose transb, int m_block, int n_block, int k_block, int m_c_tile, int n_c_tile, int m_a_tile, int n_a_tile, int m_b_tile, int n_b_tile, int use_padding>
void gemm(const concurrency::accelerator_view& av, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, value_type beta, const concurrency::array_view<value_type,2>& c)
{
    // derived tuning parameters
    static const int m_thread = m_block / m_c_tile;
    static const int n_thread = n_block / n_c_tile;

    // extract gemm sizes 
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
		tile_static value_type a_tile[m_block][k_block + a_padding];
		tile_static value_type b_tile[k_block][n_block + b_padding];

        // global tile offset indexing
		const int i = tid.tile[0] * m_block;    
		const int j = tid.tile[1] * n_block;

        // local c indexing [m_c_tile x n_c_tile]
		const int i_c_idx = tid.local[0];
		const int j_c_idx = tid.local[1]; 

        // 1D index 
        const int idx = i_c_idx * n_c_tile + j_c_idx;    

        // local a index [m_a_tile x n_a_tile]
        const int i_a_idx = idx / n_a_tile;
        const int j_a_idx = idx % n_a_tile;

        // local b index [m_b_tile x n_b_tile]
        const int i_b_idx = idx / n_b_tile;
        const int j_b_idx = idx % n_b_tile; 

        // local computation registers
        value_type c_reg[m_thread][n_thread]; 
        value_type a_reg[m_thread];
        value_type b_reg[n_thread];

        // initialize registers
        for (int m = 0; m < m_thread; m++)
             for (int n = 0; n < n_thread; n++)
                 c_reg[m][n] = value_type();

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
                        c_reg[m][n] += (transa == transpose::conj_trans ? conj(a_reg[m]) : a_reg[m]) * (transb == transpose::conj_trans ? conj(b_reg[n]) : b_reg[n]);
            }

            // wait for tiled static memory to be consumed
            tid.barrier.wait_with_tile_static_memory_fence();
        
        } // outer k-loop

        // write to c
		for (int m = 0; m < m_thread; m++)
		{
			const int m_out = i + (m*m_c_tile+i_c_idx); 
			for (int n = 0; n < n_thread; n++)
			{
				const int n_out = j + (n*n_c_tile+j_c_idx);

                value_type c_temp = guarded_read<guarded>(c, concurrency::index<2>(m_out, n_out));
				c_temp = alpha*c_reg[m][n] + beta*c_temp;
                guarded_write<guarded>(c, concurrency::index<2>(m_out, n_out), c_temp);
			}
		}
    });
}

TUNE_NAMESPACE_END

#endif // AMPBLAS_GEMM_PROFILE_GEMM_H
