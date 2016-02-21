/*----------------------------------------------------------------------------
* Copyright (c) Microsoft Corp.
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
* C++ AMP standard algorithms library.
*
* This file contains the helpers templates on which the amp_algorithms.h 
* header depends.
*---------------------------------------------------------------------------*/

#pragma once

#include <amp.h>
#include <assert.h>
#include <sstream>

#include <xx_amp_algorithms_impl_inl.h>
#include <amp_indexable_view.h>

namespace amp_algorithms
{
    // This header needs these enums but they also have to be defined in the main header.
#if !defined(AMP_ALGORITHMS_ENUMS)
#define AMP_ALGORITHMS_ENUMS
    enum class scan_mode : int
    {
        exclusive = 0,
        inclusive = 1
    };

    enum class scan_direction : int
    {
        forward = 0,
        backward = 1
    };
#endif

    namespace _details
    {
        inline concurrency::accelerator_view auto_select_target()
        {
#if _MSC_VER < 1800
            static concurrency::accelerator_view auto_select_accelerator_view = concurrency::accelerator(concurrency::accelerator::cpu_accelerator).create_view();
            return auto_select_accelerator_view;
#else
            return concurrency::accelerator::get_auto_selection_view();
#endif
        }

        //----------------------------------------------------------------------------
        // parallel_for_each implementation
        //----------------------------------------------------------------------------

        template <int _Rank, typename _Kernel_type>
        void parallel_for_each(const concurrency::accelerator_view &_Accl_view, const concurrency::extent<_Rank>& _Compute_domain, const _Kernel_type &_Kernel)
        {
#if _MSC_VER < 1800
            _Host_Scheduling_info _SchedulingInfo = { NULL };
            if (_Accl_view != _details::auto_select_target()) 
            {
                _SchedulingInfo._M_accelerator_view = concurrency::details::_Get_accelerator_view_impl_ptr(_Accl_view);
            }

            concurrency::details::_Parallel_for_each(&_SchedulingInfo, _Compute_domain, _Kernel);
#else
            concurrency::parallel_for_each(_Accl_view, _Compute_domain, _Kernel);
#endif
        }

        template <int _Dim0, int _Dim1, int _Dim2, typename _Kernel_type>
        void parallel_for_each(const concurrency::accelerator_view &_Accl_view, const concurrency::tiled_extent<_Dim0, _Dim1, _Dim2>& _Compute_domain, const _Kernel_type& _Kernel)
        {
#if _MSC_VER < 1800
            _Host_Scheduling_info _SchedulingInfo = { NULL };
            if (_Accl_view != _details::auto_select_target()) 
            {
                _SchedulingInfo._M_accelerator_view = concurrency::details::_Get_accelerator_view_impl_ptr(_Accl_view);
            }

            concurrency::details::_Parallel_for_each(&_SchedulingInfo, _Compute_domain, _Kernel);
#else
            concurrency::parallel_for_each(_Accl_view, _Compute_domain, _Kernel);
#endif
        }

        template <int _Dim0, int _Dim1, typename _Kernel_type>
        void parallel_for_each(const concurrency::accelerator_view &_Accl_view, const concurrency::tiled_extent<_Dim0, _Dim1>& _Compute_domain, const _Kernel_type& _Kernel)
        {
#if _MSC_VER < 1800
            _Host_Scheduling_info _SchedulingInfo = { NULL };
            if (_Accl_view != _details::auto_select_target()) 
            {
                _SchedulingInfo._M_accelerator_view = concurrency::details::_Get_accelerator_view_impl_ptr(_Accl_view);
            }

            concurrency::details::_Parallel_for_each(&_SchedulingInfo, _Compute_domain, _Kernel);
#else
            concurrency::parallel_for_each(_Accl_view, _Compute_domain, _Kernel);
#endif
        }

        template <int _Dim0, typename _Kernel_type>
        void parallel_for_each(const concurrency::accelerator_view &_Accl_view, const concurrency::tiled_extent<_Dim0>& _Compute_domain, const _Kernel_type& _Kernel)
        {
#if _MSC_VER < 1800
            _Host_Scheduling_info _SchedulingInfo = { NULL };
            if (_Accl_view != _details::auto_select_target()) 
            {
                _SchedulingInfo._M_accelerator_view = concurrency::details::_Get_accelerator_view_impl_ptr(_Accl_view);
            }

            concurrency::details::_Parallel_for_each(&_SchedulingInfo, _Compute_domain, _Kernel);
#else
            concurrency::parallel_for_each(_Accl_view, _Compute_domain, _Kernel);
#endif
        }

        //----------------------------------------------------------------------------
        // reduce implementation
        //---------------------------------------------------------------------------- 
        //
        // This function performs an in-place reduction through co-operating threads within a tile.
        // The input data is in the parameter "mem" and is reduced in-place modifying its existing contents
        // The output (reduced result) is contained in "mem[0]" at the end of this function
        // The parameter "partial_data_length" is used to indicate if the size of data in "mem" to be
        // reduced is same as the tile size and if not what is the length of valid data in "mem".

        template <unsigned int tile_size, typename functor, typename T>
        void reduce_tile(T* const mem, concurrency::tiled_index<tile_size> tid, const functor& op, int partial_data_length) restrict(amp)
        {
            int local = tid.local[0];

            if (partial_data_length < tile_size)
            {
                // unrolled for performance
                if (partial_data_length > 512) { if (local < (partial_data_length - 512)) { mem[0] = op(mem[0], mem[512]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 256) { if (local < (partial_data_length - 256)) { mem[0] = op(mem[0], mem[256]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 128) { if (local < (partial_data_length - 128)) { mem[0] = op(mem[0], mem[128]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 64) { if (local < (partial_data_length - 64)) { mem[0] = op(mem[0], mem[64]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 32) { if (local < (partial_data_length - 32)) { mem[0] = op(mem[0], mem[32]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 16) { if (local < (partial_data_length - 16)) { mem[0] = op(mem[0], mem[16]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 8) { if (local < (partial_data_length - 8)) { mem[0] = op(mem[0], mem[8]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 4) { if (local < (partial_data_length - 4)) { mem[0] = op(mem[0], mem[4]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 2) { if (local < (partial_data_length - 2)) { mem[0] = op(mem[0], mem[2]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (partial_data_length > 1) { if (local < (partial_data_length - 1)) { mem[0] = op(mem[0], mem[1]); } tid.barrier.wait_with_tile_static_memory_fence(); }
            }
            else
            {
                // unrolled for performance
                if (tile_size >= 1024) { if (local < 512) { mem[0] = op(mem[0], mem[512]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 512) { if (local < 256) { mem[0] = op(mem[0], mem[256]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 256) { if (local < 128) { mem[0] = op(mem[0], mem[128]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 128) { if (local < 64) { mem[0] = op(mem[0], mem[64]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 64) { if (local < 32) { mem[0] = op(mem[0], mem[32]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 32) { if (local < 16) { mem[0] = op(mem[0], mem[16]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 16) { if (local < 8) { mem[0] = op(mem[0], mem[8]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 8) { if (local < 4) { mem[0] = op(mem[0], mem[4]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 4) { if (local < 2) { mem[0] = op(mem[0], mem[2]); } tid.barrier.wait_with_tile_static_memory_fence(); }
                if (tile_size >= 2) { if (local < 1) { mem[0] = op(mem[0], mem[1]); } tid.barrier.wait_with_tile_static_memory_fence(); }
            }
        }

        // Generic reduction of a 1D indexable view with a reduction binary functor

        template<unsigned int tile_size,
            unsigned int max_tiles,
            typename InputIndexableView,
            typename BinaryFunction>
            typename std::result_of<BinaryFunction(const typename indexable_view_traits<InputIndexableView>::value_type&, const typename indexable_view_traits<InputIndexableView>::value_type&)>::type
            reduce(const concurrency::accelerator_view &accl_view, const InputIndexableView &input_view, const BinaryFunction &binary_op)
        {
            // The input view must be of rank 1
            static_assert(indexable_view_traits<InputIndexableView>::rank == 1, "The input indexable view must be of rank 1");
            typedef typename std::result_of<BinaryFunction(const typename indexable_view_traits<InputIndexableView>::value_type&, const typename indexable_view_traits<InputIndexableView>::value_type&)>::type result_type;

            // runtime sizes
            int n = input_view.extent.size();
            unsigned int tile_count = (n + tile_size - 1) / tile_size;
            tile_count = std::min(tile_count, max_tiles);

            // simultaneous live threads
            const unsigned int thread_count = tile_count * tile_size;

            // global buffer (return type)
            concurrency::array_view<result_type> global_buffer_view(concurrency::array<result_type>(tile_count, concurrency::accelerator(concurrency::accelerator::cpu_accelerator).default_view, accl_view));

            // configuration
            concurrency::extent<1> extent(thread_count);

            _details::parallel_for_each(
                accl_view,
                extent.tile<tile_size>(),
                [=](concurrency::tiled_index<tile_size> tid) restrict(amp)
            {
                // shared tile buffer
                tile_static result_type local_buffer[tile_size];

                int idx = tid.global[0];

                // this thread's shared memory pointer
                result_type& smem = local_buffer[tid.local[0]];

                // this variable is used to test if we are on the edge of data within tile
                int partial_data_length = tile_partial_data_size(input_view, tid);

                // initialize local buffer
                smem = input_view[concurrency::index<1>(idx)];
                // next chunk
                idx += thread_count;

                // fold data into local buffer
                while (idx < n)
                {
                    // reduction of smem and X[idx] with results stored in smem
                    smem = binary_op(smem, input_view[concurrency::index<1>(idx)]);

                    // next chunk
                    idx += thread_count;
                }

                // synchronize
                tid.barrier.wait_with_tile_static_memory_fence();

                // reduce all values in this tile
                _details::reduce_tile(&smem, tid, binary_op, partial_data_length);

                // only 1 thread per tile does the inter tile communication
                if (tid.local[0] == 0)
                {
                    // write to global buffer in this tiles
                    global_buffer_view[tid.tile[0]] = smem;
                }
            });

            // 2nd pass reduction
            result_type *pGlobalBufferViewData = global_buffer_view.data();
            result_type retVal = pGlobalBufferViewData[0];
            for (unsigned int i = 1; i < tile_count; ++i) {
                retVal = binary_op(retVal, pGlobalBufferViewData[i]);
            }

            return retVal;
        }

        //----------------------------------------------------------------------------
        // scan - C++ AMP implementation
        //----------------------------------------------------------------------------
        //
        // References:
        //
        // "GPU Gems 3, chapter 39. Parallel Prefix Sum (Scan) with CUDA" http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
        //
        // https://research.nvidia.com/sites/default/files/publications/nvr-2008-003.pdf
        // https://sites.google.com/site/duanemerrill/ScanTR2.pdf
        //
        // TODO: There may be some better scan implementations that are described in the second reference. Investigate.
        // TODO: Scan only supports Rank of 1.
        // TODO: Scan does not support forwards/backwards.

        static const int scan_default_tile_size = 512;

        template <int TileSize, typename _BinaryOp, typename T>
        inline T scan_tile_exclusive(T* const tile_data, concurrency::tiled_index<TileSize> tidx, const _BinaryOp& op, const int partial_data_length) restrict(amp)
        {
            const int lidx = tidx.local[0];
 
            for (int stride = 1; stride <= (TileSize / 2); stride *= 2)
            {
                if ((lidx + 1) % (stride * 2) == 0)
                {
                    tile_data[lidx] = op(tile_data[lidx], tile_data[lidx - stride]);
                }
                tidx.barrier.wait_with_tile_static_memory_fence();
            }

            if (lidx == 0)
            {
                tile_data[TileSize - 1] = 0;
            }
            tidx.barrier.wait_with_tile_static_memory_fence();

            for (int stride = TileSize / 2; stride >= 1; stride /= 2)
            {
                if ((lidx + 1) % (stride * 2) == 0)
                {
                    auto tmp = tile_data[lidx];
                    tile_data[lidx] = op(tile_data[lidx], tile_data[lidx - stride]);
                    tile_data[lidx - stride] = tmp;
                }
                tidx.barrier.wait_with_tile_static_memory_fence();
            }
            return tile_data[TileSize - 1];
        }

        template <int TileSize, scan_mode _Mode, typename _BinaryFunc, typename InputIndexableView>
        inline void scan(const concurrency::accelerator_view& accl_view, const InputIndexableView& input_view, InputIndexableView& output_view, const _BinaryFunc& op)
        {
            typedef InputIndexableView::value_type T;

            const auto compute_domain = output_view.extent.tile<TileSize>().pad();
            concurrency::array<T, 1> tile_results(compute_domain / TileSize, accl_view);
            concurrency::array_view<T, 1> tile_results_vw(tile_results);

            // 1 & 2. Scan all tiles and store results in tile_results.

            concurrency::parallel_for_each(accl_view, compute_domain, [=](concurrency::tiled_index<TileSize> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int lidx = tidx.local[0];
                const int partial_data_length = tile_partial_data_size(output_view, tidx);

                tile_static T tile_data[TileSize];
                tile_data[lidx] = (lidx >= partial_data_length) ? 0 : input_view[gidx];
                const T current_value = tile_data[lidx];
                tidx.barrier.wait_with_tile_static_memory_fence();

                auto val = _details::scan_tile_exclusive<TileSize>(tile_data, tidx, amp_algorithms::plus<T>(), partial_data_length);
                if (_Mode == scan_mode::inclusive)
                {
                    tile_data[lidx] += current_value;
                }

                if (lidx == (TileSize - 1))
                {
                    tile_results_vw[tidx.tile[0]] = val + current_value;
                }
                padded_write(output_view, gidx, tile_data[lidx]);
            });

            // 3. Scan tile results.

            if (tile_results_vw.extent[0] > TileSize)
            {
                scan<TileSize, amp_algorithms::scan_mode::exclusive>(accl_view, tile_results_vw, tile_results_vw, op);
            }
            else
            {
                concurrency::parallel_for_each(accl_view, compute_domain, [=](concurrency::tiled_index<TileSize> tidx) restrict(amp)
                {
                    const int gidx = tidx.global[0];
                    const int lidx = tidx.local[0];
                    const int partial_data_length = tile_partial_data_size(tile_results_vw, tidx);

                    tile_static T tile_data[TileSize];
                    tile_data[lidx] = tile_results_vw[gidx];
                    tidx.barrier.wait_with_tile_static_memory_fence();

                    _details::scan_tile_exclusive<TileSize>(tile_data, tidx, amp_algorithms::plus<T>(), partial_data_length);

                    tile_results_vw[gidx] = tile_data[lidx];
                    tidx.barrier.wait_with_tile_static_memory_fence();
                });
            }

            // 4. Add the tile results to the individual results for each tile.

            concurrency::parallel_for_each(accl_view, compute_domain, [=](concurrency::tiled_index<TileSize> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];

                if (gidx < output_view.extent[0])
                {
                    output_view[gidx] += tile_results_vw[tidx.tile[0]];
                }
            });
        }

        // This takes an exclusive scan result and calculates the equivalent of a segmented scan at element i for a regular segment_width.
        // It is used internally to do segmented scans without the need to use a heavier segmented scan with arbitrary segment widths.

        template <typename T>
        inline T segment_exclusive_scan(const array_view<T, 1>& exclusive_scan, const int segment_width, const int i) restrict(amp, cpu)
        {
            return exclusive_scan[i] - exclusive_scan[i - (i % segment_width)];
        }

        //----------------------------------------------------------------------------
        // segmented scan - C++ AMP implementation
        //----------------------------------------------------------------------------
        //
        // References:
        //
        // "Efficient Parallel Scan Algorithms for GPUs" http://www.gpucomputing.net/sites/default/files/papers/2590/nvr-2008-003.pdf

        template <int TileSize, scan_mode _Mode, typename _BinaryFunc, typename InputIndexableView>
        inline void segmented_scan(const concurrency::accelerator_view& accl_view, const InputIndexableView& input_view, InputIndexableView& output_view, const _BinaryFunc& op)
        {
            // TODO_NOT_IMPLEMENTED: segmented_scan.
        }

        //----------------------------------------------------------------------------
        // radix sort implementation
        //----------------------------------------------------------------------------
        //
        // References:
        //
        // "Introduction to GPU Radix Sort" http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf
        //
        // "Designing Efficient Sorting Algorithms for Manycore GPUs" http://www.nvidia.com/docs/io/67073/nvr-2008-001.pdf
        // "Histogram Calculation in CUDA" http://docs.nvidia.com/cuda/samples/3_Imaging/histogram/doc/histogram.pdf
        
        template<typename T, int key_bit_width>
        inline int radix_key_value(const T value, const unsigned key_idx) restrict(amp, cpu)
        {
            const T mask = (1 << key_bit_width) - 1;
            return (value >> (key_idx * key_bit_width)) & mask;
        }

        template <typename T>
        inline void initialize_bins(T* const bin_data, const int bin_count) restrict(amp)
        {
            for (int b = 0; b < bin_count; ++b)
            {
                bin_data[b] = T(0);
            }
        }

        template <typename T, int tile_size, int tile_key_bit_width>
        void radix_sort_tile_by_key(T* const tile_data, const int data_size, concurrency::tiled_index<tile_size> tidx, const int key_idx) restrict(amp)
        {
            const unsigned bin_count = 1 << tile_key_bit_width;
            const int gidx = tidx.global[0];
            const int tlx = tidx.tile[0];
            const int idx = tidx.local[0];

            // Increment histogram bins for each element.

            tile_static unsigned long tile_radix_values[tile_size];
            tile_radix_values[idx] = pack_byte(1, _details::radix_key_value<T, tile_key_bit_width>(tile_data[idx], key_idx));
            tidx.barrier.wait_with_tile_static_memory_fence();

            tile_static unsigned long histogram_bins_scan[bin_count];
            if (idx == 0)
            {
                // Calculate histogram of radix values. Don't add values that are off the end of the data.
                unsigned long global_histogram = 0;
                const int tile_data_size = amp_algorithms::min<int>()(tile_size, (data_size - (tlx * tile_size)));
                for (int i = 0; i < tile_data_size; ++i)
                {
                    global_histogram += tile_radix_values[i];
                }

                // Scan to get offsets for each histogram bin.

                histogram_bins_scan[0] = 0;
                for (int i = 1; i < bin_count; ++i)
                {
                    histogram_bins_scan[i] = unpack_byte(global_histogram, i - 1) + histogram_bins_scan[i - 1];
                }
            }
            tidx.barrier.wait_with_tile_static_memory_fence();

            _details::scan_tile_exclusive<tile_size>(tile_radix_values, tidx, amp_algorithms::plus<unsigned long>(), tile_size);

            // Shuffle data into sorted order.

            T tmp = tile_data[idx];
            tidx.barrier.wait_with_tile_static_memory_fence();
            if (gidx < data_size)
            {
                const int rdx = _details::radix_key_value<T, tile_key_bit_width>(tmp, key_idx);
                unsigned long dest_idx = histogram_bins_scan[rdx] + unpack_byte(tile_radix_values[idx], rdx);
                tile_data[dest_idx] = tmp;
            }
        }

        template <typename T, int tile_size, int key_bit_width, int tile_key_bit_width = 2>
        void radix_sort_by_key(const concurrency::accelerator_view& accl_view, const concurrency::array_view<T>& input_view, concurrency::array_view<T>& output_view, const int key_idx)
        {
            static const unsigned type_width = sizeof(T) * CHAR_BIT;
            static const int bin_count = 1 << key_bit_width;

            static_assert((tile_size <= 256), "The tile size must be less than or equal to 256.");
            static_assert((key_bit_width >= 1), "The radix bit width must be greater than or equal to one.");
            static_assert((tile_size >= bin_count), "The tile size must be greater than or equal to the radix key bin count.");
            static_assert((type_width % key_bit_width == 0), "The sort key width must be divisible by the type width.");
            static_assert((key_bit_width % tile_key_bit_width == 0), "The key bit width must be divisible by the tile key bit width.");
            static_assert(tile_key_bit_width <= 2, "Only tile key bin widths of two or less are supported.");

            const concurrency::tiled_extent<tile_size> compute_domain = output_view.get_extent().tile<tile_size>().pad();
            const int tile_count = std::max(1u, compute_domain.size() / tile_size);

            concurrency::array<int, 2> per_tile_rdx_offsets(concurrency::extent<2>(tile_count, bin_count), accl_view);
            concurrency::array<int> global_rdx_offsets(bin_count, accl_view);
            concurrency::array<int, 1> tile_histograms(concurrency::extent<1>(bin_count * tile_count), accl_view);

            amp_algorithms::fill(accl_view, global_rdx_offsets.section(0, bin_count), 0);

            concurrency::parallel_for_each(accl_view, compute_domain, [=, &per_tile_rdx_offsets, &global_rdx_offsets, &tile_histograms](concurrency::tiled_index<tile_size> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int tlx = tidx.tile[0];
                const int idx = tidx.local[0];
                tile_static T tile_data[tile_size];
                tile_static int per_thread_rdx_histograms[tile_size][bin_count];

                // Initialize histogram bins and copy data into tiles.
                initialize_bins(per_thread_rdx_histograms[idx], bin_count);
                tile_data[idx] = padded_read(input_view, gidx);

                // Increment radix bins for each element on each tile.
                if (gidx < input_view.extent[0])
                {
                    per_thread_rdx_histograms[idx][_details::radix_key_value<T, key_bit_width>(tile_data[idx], key_idx)]++;
                }
                tidx.barrier.wait_with_tile_static_memory_fence();

                // First bin_count threads per tile collapse thread values to create the tile histogram.
                if (idx < bin_count)
                {
                    for (int i = 1; i < tile_size; ++i)
                    {
                        per_thread_rdx_histograms[0][idx] += per_thread_rdx_histograms[i][idx];
                    }
                }
                tidx.barrier.wait_with_tile_static_memory_fence();

                // First bin_count threads per tile increment counts for global histogram and copies tile histograms to global memory.
                if (idx < bin_count)
                {
                    concurrency::atomic_fetch_add(&global_rdx_offsets[idx], per_thread_rdx_histograms[0][idx]);
                }

                //output_view[gidx] = (idx < bin_count) ? per_thread_rdx_histograms[0][idx] : 0;            // Dump per-tile histograms, per_tile_rdx_histograms

                // Exclusive scan the tile histogram to calculate the per-tile offsets.
                if (idx < bin_count)
                {
                    tile_histograms[(idx * tile_count) + tlx] = per_thread_rdx_histograms[0][idx];
                }
                tidx.barrier.wait_with_tile_static_memory_fence();
                _details::scan_tile_exclusive<tile_size>(per_thread_rdx_histograms[0], tidx, amp_algorithms::plus<T>(), tile_size);

                if (idx < bin_count)
                {
                    per_tile_rdx_offsets[tlx][idx] = per_thread_rdx_histograms[0][idx];
                }
            });

            concurrency::parallel_for_each(accl_view, compute_domain, [=, &global_rdx_offsets, &tile_histograms](concurrency::tiled_index<tile_size> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int idx = tidx.local[0];

                //output_view[gidx] = (gidx < bin_count) ? global_rdx_offsets[gidx] : 0;                    // Dump per-tile histograms, global_histogram
                //output_view[gidx] = (gidx < bin_count * tile_count) ? tile_histograms[gidx] : 0;          // Dump per-tile histograms, per_tile_rdx_histograms_tp,

                // Calculate global radix offsets from the global radix histogram. All tiles do this but only the first one records the result.
                tile_static int scan_data[tile_size];
                scan_data[idx] = (idx < bin_count) ? global_rdx_offsets[idx] : 0;
                tidx.barrier.wait_with_tile_static_memory_fence();

                _details::scan_tile_exclusive<tile_size>(scan_data, tidx, amp_algorithms::plus<T>(), tile_size);

                if (gidx < bin_count)
                {
                    global_rdx_offsets[gidx] = scan_data[gidx];
                }
            });

            concurrency::array_view<int, 1> tile_histograms_vw(tile_histograms);
            scan_exclusive(tile_histograms_vw, tile_histograms_vw);

            concurrency::parallel_for_each(accl_view, compute_domain, [=, &per_tile_rdx_offsets, &tile_histograms, &global_rdx_offsets](concurrency::tiled_index<tile_size> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int tlx = tidx.tile[0];
                const int idx = tidx.local[0];

                // Check inputs from previous steps are correct:
                //
                //if (idx < bin_count) { output_view[gidx] = tile_histograms[(idx * tile_count) + tlx]; }                                               // Dump tile offsets, tile_rdx_offsets
                //if (idx < bin_count) { output_view[gidx] = per_tile_rdx_offsets[tlx][idx]; }                                                          // Dump per tile offsets, per_tile_rdx_offsets
                //if (idx < bin_count * tile_count) { output_view[gidx] = segment_exclusive_scan(tile_histograms_vw, tile_count, gidx); }               // Dump tile offsets, tile_histogram_segscan
                //output_view[gidx] = (gidx < bin_count) ? global_rdx_offsets[gidx] : 0;                                                                // Dump global offsets, global_rdx_offsets

                // Sort elements within each tile.
                tile_static T tile_data[tile_size];
                tile_data[idx] = input_view[gidx];
                tidx.barrier.wait_with_tile_static_memory_fence();

                const int keys_per_tile = (key_bit_width / tile_key_bit_width);
                for (int k = (keys_per_tile * key_idx); k < (keys_per_tile * (key_idx + 1)); ++k)
                {
                    _details::radix_sort_tile_by_key<T, tile_size, tile_key_bit_width>(tile_data, input_view.extent.size(), tidx, k);
                }
                tidx.barrier.wait_with_tile_static_memory_fence();

                //output_view[gidx] = tile_data[idx];                                                       // Dump sorted per-tile data, sorted_per_tile

                // Move tile sorted elements to global destination.

                const int rdx = _details::radix_key_value<T, key_bit_width>(tile_data[idx], key_idx);
                const int dest_gidx = 
                    idx - 
                    per_tile_rdx_offsets[tlx][rdx] + 
                    segment_exclusive_scan(tile_histograms_vw, tile_count, (rdx * tile_count) + tlx) +
                    global_rdx_offsets[rdx];

                //output_view[gidx] = dest_gidx;                                                            // Dump destination indices, dest_gidx

                output_view[dest_gidx] = tile_data[idx];
            });
        }

        template <typename T, int tile_size, int key_bit_width, int tile_key_bit_width = 2>
        void radix_sort(const concurrency::accelerator_view& accl_view, concurrency::array_view<T>& input_view, concurrency::array_view<T>& output_view)
        {
            static const int key_count = bit_count<T>() / key_bit_width;

            for (int key_idx = 0; key_idx < key_count; ++key_idx)
            {
                _details::radix_sort_by_key<T, tile_size, key_bit_width, tile_key_bit_width>(accl_view, input_view, output_view, key_idx);
                std::swap(output_view, input_view);
            }
            std::swap(input_view, output_view);
        }

    } // namespace amp_algorithms::_details

} // namespace amp_algorithms
