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
 * reduction.h 
 *
 * High performance reduction algorithm
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_UTILITY_REDUCTION_H
#define AMPBLAS_UTILITY_REDUCTION_H

#include "ampblas_config.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

template <typename T, unsigned int tile_size, typename functor>
void tile_local_reduction(T* const mem, concurrency::tiled_index<tile_size> tid, const functor& op) restrict(amp)
{
    // local index
    unsigned int local = tid.local[0];

    // unrolled for performance
    if (tile_size >= 1024) { if (local < 512) { mem[0] = op(mem[0], mem[512]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=  512) { if (local < 256) { mem[0] = op(mem[0], mem[256]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=  256) { if (local < 128) { mem[0] = op(mem[0], mem[128]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=  128) { if (local <  64) { mem[0] = op(mem[0], mem[ 64]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=   64) { if (local <  32) { mem[0] = op(mem[0], mem[ 32]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=   32) { if (local <  16) { mem[0] = op(mem[0], mem[ 16]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=   16) { if (local <   8) { mem[0] = op(mem[0], mem[  8]); } tid.barrier.wait_with_tile_static_memory_fence(); }   
    if (tile_size >=    8) { if (local <   4) { mem[0] = op(mem[0], mem[  4]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=    4) { if (local <   2) { mem[0] = op(mem[0], mem[  2]); } tid.barrier.wait_with_tile_static_memory_fence(); }
    if (tile_size >=    2) { if (local <   1) { mem[0] = op(mem[0], mem[  1]); } tid.barrier.wait_with_tile_static_memory_fence(); }
}

// Generic reduction of an 1D container with the reduction operation specified by a helper functor
template <unsigned int tile_size, unsigned int max_tiles, typename ret_type, typename elm_type, typename x_type, typename functor>
ret_type reduce(const concurrency::accelerator_view& av, int n, const x_type& X, const functor& reduce_helper)
{
    // runtime sizes
    unsigned int tile_count = (n+tile_size-1) / tile_size;
    tile_count = std::min(tile_count, max_tiles);   

    // simultaneous live threads
    const unsigned int thread_count = tile_count * tile_size;

    // global buffer (return type)
    concurrency::array<elm_type,1> global_buffer(tile_count);
    concurrency::array_view<elm_type,1> global_buffer_view(global_buffer);

    // configuration
    concurrency::extent<1> extent(thread_count);

    concurrency::parallel_for_each(
        av,
        extent.tile<tile_size>(),
        [=] (concurrency::tiled_index<tile_size> tid) restrict(amp)
    {
        // shared tile buffer
        tile_static elm_type local_buffer[tile_size];

        // indexes
        int idx = tid.global[0];

        // this threads's shared memory pointer
        elm_type& smem = local_buffer[ tid.local[0] ];

        // initialize local buffer
        smem = reduce_helper.init_value;

        // fold data into local buffer
        while (idx < n)
        {
            // reduction of smem and X[idx] with results stored in smem
            reduce_helper.local_reduce(smem, idx, X);

            // next chunk
            idx += thread_count;
        }

        // synchronize
        tid.barrier.wait_with_tile_static_memory_fence();

        // reduce all values in this tile
        _detail::tile_local_reduction<elm_type,tile_size>(&smem, tid, reduce_helper.op);

        // only 1 thread per tile does the inter tile communication
        if (tid.local[0] == 0)
        {
            // write to global buffer in this tiles
            global_buffer_view[ tid.tile[0] ] = smem;
        }
    });

    // 2nd pass reduction
    std::vector<elm_type> host_buffer(global_buffer);
    return reduce_helper.global_reduce(host_buffer);
}

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_UTILITY_REDUCTION_H
