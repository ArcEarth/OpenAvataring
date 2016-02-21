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
* C++ AMP standard algorithm library.
*
* This file contains the implementation for C++ AMP standard algorithms
*---------------------------------------------------------------------------*/

#pragma once

#include <functional>
#include <numeric>

#include <amp_stl_algorithms.h>
#include <amp_algorithms.h>

namespace amp_stl_algorithms
{
    namespace _details
    {
        template<class ConstRandomAccessIterator>
        concurrency::array_view<typename std::iterator_traits<ConstRandomAccessIterator>::value_type> 
            create_section(ConstRandomAccessIterator iter, typename std::iterator_traits<ConstRandomAccessIterator>::difference_type distance) 
        {
            typedef std::iterator_traits<ConstRandomAccessIterator>::value_type value_type;
            typedef std::iterator_traits<ConstRandomAccessIterator>::difference_type difference_type;
            auto base_view = _details::array_view_iterator_helper<value_type>::get_base_array_view(iter);
            difference_type start = std::distance(begin(base_view), iter);
            return base_view.section(concurrency::index<1>(start), concurrency::extent<1>(distance));
        }		
    }

    // TODO: Get the tests, header and internal implementations into the same logical order.
    // TODO: Lots of the algorithms that typically do a small amount of work per thread should use tiling to save the runtime overhead of having to do this. 

    //----------------------------------------------------------------------------
    // adjacent_difference
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename BinaryOperation>
    RandomAccessIterator adjacent_difference( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first,
        BinaryOperation p )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::value_type T;

        const diff_type element_count = std::distance(first, last);
        if (element_count <= 1)
        {
            return dest_first;
        }

        static const int tile_size = 512;
        auto input_view = _details::create_section(first, element_count);
        auto output_view = _details::create_section(dest_first, element_count);

        concurrency::tiled_extent<tile_size> compute_domain = concurrency::extent<1>(element_count).tile<tile_size>().pad();
        concurrency::parallel_for_each(compute_domain, [=](concurrency::tiled_index<tile_size> tidx) restrict(amp)
        {
            const int idx = tidx.global[0];
            const int i = tidx.local[0];
            tile_static T local_buffer[tile_size + 1];

            local_buffer[i] = padded_read(input_view, idx - 1);
            if (i == (tile_size - 1))
            {
                local_buffer[tile_size] = padded_read(input_view, idx);
            }

            tidx.barrier.wait_with_all_memory_fence();
             
            if (idx == 0)
            {
                output_view[0] = input_view[0];
            }
            else
            {
                padded_write(output_view, idx, p(local_buffer[i + 1], local_buffer[i]));
            }
        });

        return dest_first + element_count;
    }

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator>
    RandomAccessIterator adjacent_difference( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::value_type T;
        return amp_stl_algorithms::adjacent_difference(first, last, dest_first, amp_algorithms::minus<T>());
    }

    //----------------------------------------------------------------------------
    // all_of, any_of, none_of
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator,  typename UnaryPredicate>
    bool all_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p ) 
    {
        return !amp_stl_algorithms::any_of(
            first, last, 
            [p] (const decltype(*first)& val) restrict(amp) { return !p(val); }
        );
    }

    // Non-standard, OutputIterator must yield an int reference, where the result will be
    // stored. This allows the function to eschew synchronization
    template<typename ConstRandomAccessIterator, typename UnaryPredicate, typename OutputIterator>
    void any_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p, OutputIterator dest_first)
    {
        auto section_view = _details::create_section(dest_first, 1);
        section_view[0] = 0;
        amp_stl_algorithms::for_each_no_return(
            first, last, 
            [section_view, p] (const decltype(*first)& val) restrict(amp) 
        {
            int *accumulator = &section_view(0);
            if (*accumulator == 0)
            {
                if (p(val))
                {
                    concurrency::atomic_exchange(accumulator, 1);
                }
            }
        });
    }

    // Standard, builds of top of the non-standard async version above, and adds a sync to
    // materialize the result.
    template<typename ConstRandomAccessIterator, typename UnaryPredicate >
    bool any_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p )
    {
        int found_any_storage = 0;
        concurrency::array_view<int> found_any_av(concurrency::extent<1>(1), &found_any_storage);
        amp_stl_algorithms::any_of(first, last, p, amp_stl_algorithms::begin(found_any_av));
        found_any_av.synchronize();
        return found_any_storage == 1;
    }

    template<typename ConstRandomAccessIterator, typename UnaryPredicate >
    bool none_of( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p )
    {
        return !amp_stl_algorithms::any_of(first, last, p);
    }

    //----------------------------------------------------------------------------
    // copy, copy_if, copy_n
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator copy( ConstRandomAccessIterator first,  ConstRandomAccessIterator last, RandomAccessIterator dest_first )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;
        const diff_type element_count = std::distance(first, last);
        if (element_count <= 0)
            return dest_first;
        auto src_view = _details::create_section(first, element_count);
        concurrency::copy(src_view, dest_first);
        return dest_first + element_count;
    }

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator copy_if(ConstRandomAccessIterator first,  
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        UnaryPredicate pred)
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;

        static const int tile_size = 512;
        const diff_type element_count = std::distance(first, last);
        if (element_count <= 0)
        {
            return dest_first;
        }
        auto src_view = _details::create_section(first, element_count);
        auto dest_view = _details::create_section(dest_first, element_count);

        auto map_size = concurrency::extent<1>(element_count + 1).tile<tile_size>().pad().size();
        concurrency::array<unsigned int> map(map_size);
        concurrency::array_view<unsigned int> map_vw(map);
        concurrency::tiled_extent<tile_size> compute_domain = concurrency::extent<1>(element_count).tile<tile_size>().pad();
        concurrency::parallel_for_each(compute_domain,
            [src_view, map_vw, pred, element_count](concurrency::tiled_index<tile_size> tidx) restrict(amp)
        {
            const int idx = tidx.global[0];
            map_vw[idx] = (idx < element_count) ? static_cast<unsigned int>(pred(src_view[idx])) : 0;
        });

        std::vector<unsigned int> map_dbg(map_size);
        concurrency::copy(map, begin(map_dbg));

        amp_algorithms::scan_exclusive(map_vw, map_vw);
        concurrency::copy(map, begin(map_dbg));

        dest_view.discard_data();
        concurrency::parallel_for_each(compute_domain, [=](concurrency::tiled_index<tile_size> tidx) restrict(amp)
        {
            const int idx = tidx.global[0];
            const int i = tidx.local[0];
            tile_static unsigned local_buffer[tile_size + 1];

            // Use tile memory so that each value is only read once from global memory.
            local_buffer[i] = map_vw[idx];
            if (i == (tile_size - 1))
            {
                local_buffer[i + 1] = map_vw[idx + 1];
            }

            tidx.barrier.wait_with_all_memory_fence();

            if ((local_buffer[i] != local_buffer[i + 1]) && (idx < element_count))
            {
                dest_view[map_vw[idx]] = src_view[idx];
            }
        });

        int remaining_elements;
        concurrency::copy(map_vw.section(element_count, 1), stdext::make_checked_array_iterator(&remaining_elements, 1));
        return dest_first + remaining_elements;
    }

    template<typename ConstRandomAccessIterator, typename Size, typename RandomAccessIterator>
    RandomAccessIterator copy_n(ConstRandomAccessIterator first, Size count, RandomAccessIterator dest_first)
    {
        // copy() will handle the case where count == 0.
        return amp_stl_algorithms::copy(first, (first + count), dest_first);
    }

    //----------------------------------------------------------------------------
    // count, count_if
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename T >
    typename std::iterator_traits<ConstRandomAccessIterator>::difference_type
        count( ConstRandomAccessIterator first, ConstRandomAccessIterator last, const T &value )
    {
        return amp_stl_algorithms::count_if(
            first, last, 
            [value] (const decltype(*first)& cur_val) restrict(amp) { return cur_val==value; }
        );
    }

    template<typename ConstRandomAccessIterator, typename UnaryPredicate >
    typename std::iterator_traits<ConstRandomAccessIterator>::difference_type
        count_if( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;
        const diff_type element_count = std::distance(first, last);
        if (element_count <= 0)
        {
            return 0;
        }

        diff_type count = 0;
        concurrency::array_view<diff_type> count_av(1, &count);

        auto section_view = _details::create_section(first, element_count);

        // TODO: Seems the global memory access isn't coherent. Can it be improved.
        // TODO: Would a reduction be more efficient than using an atomic operation here?
        const int num_threads = std::min(element_count, 10 * 1024);
        concurrency::parallel_for_each(concurrency::extent<1>(num_threads), 
            [num_threads, section_view, element_count, p, count_av] (concurrency::index<1> idx) restrict (amp) 
        {
            int tid = idx[0];
            diff_type local_count = 0;
            for (diff_type i = tid; i < element_count; i += num_threads)
            {
                if (p(section_view(i)))
                {
                    local_count++;
                }
            }
            if (local_count > 0)
            {
                concurrency::atomic_fetch_add(&count_av(0), local_count);
            }
        }
        );

        count_av.synchronize();
        return count;
    }

    //----------------------------------------------------------------------------
    // equal, equal_range
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename BinaryPredicate>
    bool equal( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1, 
        ConstRandomAccessIterator2 first2, 
        BinaryPredicate p )
    {
        typedef std::iterator_traits<ConstRandomAccessIterator1>::difference_type diff_type;
        diff_type element_count = std::distance(first1, last1);
        if (element_count <= 0) 
        {
            return true;
        }

        const int tile_size = 512;
        auto section1_view = _details::create_section(first1, element_count);
        auto section2_view = _details::create_section(first2, element_count);

        diff_type unequal_count = 0;
        concurrency::array_view<diff_type> unequal_count_av(1, &unequal_count);

        // TODO: Seems the global memory access isn't coherent. Can it be improved.
        // TODO: Would a reduction be more efficient than using an atomic operation here?
        const int num_threads = std::min(element_count, 10 * 1024);
        concurrency::parallel_for_each(tiled_extent<tile_size>(concurrency::extent<1>(num_threads)).pad(),
            [num_threads, section1_view, section2_view, element_count, p, unequal_count_av] (concurrency::tiled_index<tile_size> tidx) restrict (amp) 
        {
            int idx = tidx.global[0];
            diff_type i = idx;
            for (; i < element_count; i += num_threads)
            {
                if (!p(section1_view[i], section2_view[i]))
                {
                    break;
                }
            }
            if (i < element_count)
            {
                concurrency::atomic_fetch_add(&unequal_count_av(0), 1);
            }
        });
        return (unequal_count_av[0] == 0);
    }

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    bool equal( ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1, ConstRandomAccessIterator2 first2 )
    {
        typedef std::iterator_traits<ConstRandomAccessIterator1>::value_type T;

        return amp_stl_algorithms::equal(first1, last1, first2, [=](const T& v1, const T& v2) restrict(amp) { return (v1 == v2); });
    }

    //----------------------------------------------------------------------------
    // fill, fill_n
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    void fill( RandomAccessIterator first, RandomAccessIterator last, const T& value )
    {
        amp_stl_algorithms::generate(first, last, [value]() restrict(amp) { return value; });
    }

    template<typename RandomAccessIterator, typename Size, typename T>
    void fill_n( RandomAccessIterator first, Size count, const T& value )
    {
        amp_stl_algorithms::generate_n(first, count, [value]() restrict(amp) { return value; });
    }

    //----------------------------------------------------------------------------
    // find, find_if, find_if_not, find_end, find_first_of, adjacent_find
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    ConstRandomAccessIterator find_if(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p )
    {
        typedef std::iterator_traits<ConstRandomAccessIterator>::difference_type difference_type;

        difference_type element_count = std::distance(first, last);
        if (element_count <= 0) 
        {
            return last;
        }

        difference_type result_position = element_count;
        concurrency::array_view<int> result_position_av(concurrency::extent<1>(1), &result_position);

        auto section_view = _details::create_section(first, element_count);

        concurrency::parallel_for_each(concurrency::extent<1>(element_count), [=] (concurrency::index<1> idx) restrict(amp) {
            int i = idx[0];
            if (p(section_view[idx]))
            {
                concurrency::atomic_fetch_min(&result_position_av(0), i);
            }
        });

        result_position_av.synchronize();
        return first + result_position;
    }

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    ConstRandomAccessIterator find_if_not( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p )
    {
        typedef std::iterator_traits<ConstRandomAccessIterator>::value_type T;
        return amp_stl_algorithms::find_if(first, last, [p](const T& v) restrict(amp) { return !p(v); });
    }

    template<typename ConstRandomAccessIterator, typename T>
    ConstRandomAccessIterator find( ConstRandomAccessIterator first, ConstRandomAccessIterator last, const T& value )
    {
        return amp_stl_algorithms::find_if(first, last, [=] (const decltype(*first)& curr_val) restrict(amp) {
            return curr_val == value;
        });
    }

    namespace _details
    {
        template<typename ConstRandomAccessIterator, typename Predicate>
        ConstRandomAccessIterator adjacent_find (ConstRandomAccessIterator first, 
            const typename std::iterator_traits<ConstRandomAccessIterator>::difference_type element_count, Predicate pred)
        {
            static const int tile_size = 512;

            auto input_view = _details::create_section(first, element_count);
            // TODO: Here we have one idx. Might be better to have one index per tile and then do a further reduction?
            int last_sorted_idx = element_count;
            concurrency::array_view<int> last_sorted_idx_av(1, &last_sorted_idx);

            concurrency::tiled_extent<tile_size> compute_domain = concurrency::extent<1>(element_count).tile<tile_size>().pad();
            concurrency::parallel_for_each(compute_domain, [=](concurrency::tiled_index<tile_size> tidx) restrict(amp)
            {
                const int idx = tidx.global[0];
                const int i = tidx.local[0];
                tile_static unsigned local_buffer[tile_size + 1];

                local_buffer[i] = padded_read(input_view, idx);
                if (i == (tile_size - 1))
                {
                    local_buffer[i + 1] = padded_read(input_view, (idx + 1));
                }

                tidx.barrier.wait_with_all_memory_fence();

                if ((idx < element_count) && pred(local_buffer[i], local_buffer[i + 1]))
                {
                    concurrency::atomic_fetch_min(&last_sorted_idx_av(0), idx);
                }
            });

            last_sorted_idx_av.synchronize();
            return first + last_sorted_idx_av[0];
        }
    }; // namespace _details

    template<typename ConstRandomAccessIterator, typename Predicate>
    ConstRandomAccessIterator adjacent_find (ConstRandomAccessIterator first,  ConstRandomAccessIterator last, Predicate p)
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;

        const diff_type element_count = std::distance(first, last);
        if (element_count <= 1)
        {
            return last;
        }
        return  _details::adjacent_find(first, element_count, p);
    }

    template<typename ConstRandomAccessIterator>
    ConstRandomAccessIterator adjacent_find (ConstRandomAccessIterator first, ConstRandomAccessIterator last)
    {
        typedef std::iterator_traits<ConstRandomAccessIterator>::value_type T;
        return amp_stl_algorithms::adjacent_find(first, last, amp_algorithms::equal_to<T>());
    }

    //----------------------------------------------------------------------------
    // for_each, for_each_no_return
    //----------------------------------------------------------------------------

    template< typename ConstRandomAccessIterator, typename UnaryFunction >
    void for_each_no_return( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryFunction f )
    {	
        typedef std::iterator_traits<ConstRandomAccessIterator>::difference_type difference_type;

        difference_type element_count = std::distance(first, last);
        if (element_count <= 0) 
        {
            return;
        }

        auto section_view = _details::create_section(first, element_count);
        concurrency::parallel_for_each(concurrency::extent<1>(element_count), [f,section_view] (concurrency::index<1> idx) restrict(amp)
        {
            f(section_view[idx]);
        });
    }

    // UnaryFunction CANNOT contain any array, array_view or textures. Needs to be blittable.
    template< typename ConstRandomAccessIterator, typename UnaryFunction >
    UnaryFunction for_each( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryFunction f )
    {
        concurrency::array_view<UnaryFunction> functor_av(concurrency::extent<1>(1), &f);
        for_each_no_return( 
            first, last, 
            [functor_av] (const decltype(*first)& val) restrict (amp)
        {
            functor_av(0)(val);
        }
        );
        functor_av.synchronize();
        return f;
    }

    //----------------------------------------------------------------------------
    // generate, generate_n
    //
    // The "Generator" functor needs to be callable as "g()" and must return a type
    // that is assignable to RandomAccessIterator::value_type.  The functor needs
    // to be blittable and cannot contain any array, array_view, or textures.
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename Size, typename Generator>
    void generate_n(RandomAccessIterator first, Size count, Generator g)
    {
        if (count <= 0) 
        {
            return;
        }
        auto section_view = _details::create_section(first, count);
        concurrency::parallel_for_each(section_view.extent, [g,section_view] (concurrency::index<1> idx) restrict(amp) 
        {
            section_view[idx] = g();
        });
    }

    template <typename RandomAccessIterator, typename Generator>
    void generate(RandomAccessIterator first, RandomAccessIterator last, Generator g)
    {
        typedef std::iterator_traits<RandomAccessIterator>::difference_type difference_type;
        difference_type element_count = std::distance(first, last);
        amp_stl_algorithms::generate_n(first, element_count, g);
    }

    //----------------------------------------------------------------------------
    // includes
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // inner_product
    //----------------------------------------------------------------------------
    
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename T, typename BinaryOperation1, typename BinaryOperation2>
    T inner_product(ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1, ConstRandomAccessIterator2 first2, const T value,
        const BinaryOperation1& binary_op1, const BinaryOperation2& binary_op2)
    {
        typedef std::iterator_traits<ConstRandomAccessIterator1>::difference_type difference_type;
        difference_type element_count = std::distance(first1, last1);
        if (element_count <= 0)
        {
            return value;
        }
        auto section1_view = _details::create_section(first1, element_count);
        auto section2_view = _details::create_section(first2, element_count);

        concurrency::array<T> map(element_count);
        concurrency::array_view<T> map_vw(map);
        map_vw.discard_data();
        concurrency::parallel_for_each(map_vw.extent, [=](concurrency::index<1> idx) restrict(amp)
        {
            map_vw[idx] = binary_op2(section1_view[idx], section2_view[idx]);
        });
        map_vw.synchronize();
        return binary_op1(value, amp_algorithms::reduce(map_vw, binary_op1));
    }

    //----------------------------------------------------------------------------
    // iota
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    void iota( RandomAccessIterator first, RandomAccessIterator last, T value)
    {
        typedef std::iterator_traits<RandomAccessIterator>::difference_type difference_type;
        const int tile_size = 512;

        difference_type element_count = std::distance(first, last);
        if (element_count <= 0) 
        {
            return;
        }
        // TODO: This looks like it is in violation with the standard, which requires that the value needs to support 
        //       only ++value. i.e. the default ctor or subtraction, multiplication might not be available. We should 
        //       consider if there is a better alternative.
        auto inc = T();
        inc = ++inc - T();
        auto section_view = _details::create_section(first, element_count);

        concurrency::parallel_for_each(concurrency::tiled_extent<tile_size>(section_view.extent).pad(), [=](concurrency::tiled_index<tile_size> tidx) restrict(amp)
        {
            int idx = tidx.global[0];
            section_view[idx] = value + (T(idx) * inc);  // Hum... Is this numerically equivalent to incrementing?
        });
    }

    //----------------------------------------------------------------------------
    // lexographical_compare
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // lower_bound, upper_bound
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // merge, inplace_merge
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // max_element, min_element, minmax_element
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // mismatch
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // move, move_backward
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // nth_element
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // partial sum
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // partition, stable_partition, partition_point, is_partitioned
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // reduce
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename T, typename BinaryOperation>
    T reduce( ConstRandomAccessIterator first, ConstRandomAccessIterator last, T initial_value, BinaryOperation op )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;
        diff_type element_count = std::distance(first, last);
        auto section_view = _details::create_section(first, element_count);

        return op(initial_value, amp_algorithms::reduce(section_view, op));
    }

    template<typename ConstRandomAccessIterator, typename T>
    T reduce( ConstRandomAccessIterator first, ConstRandomAccessIterator last, T initial_value )
    {
        return amp_stl_algorithms::reduce(first, last, initial_value, amp_algorithms::plus<std::remove_const<typename std::iterator_traits<ConstRandomAccessIterator>::value_type>::type>());
    }

    //----------------------------------------------------------------------------
    // remove, remove_if, remove_copy, remove_copy_if
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    RandomAccessIterator remove( RandomAccessIterator first, RandomAccessIterator last, const T& value )
    {
        return amp_stl_algorithms::remove_if(first, last, [=](const T& v) restrict(amp) { return (v == value) ? 1 : 0; });
    }

    // TODO: Is is possible to remove elements in place and save having to copy?
    template<typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator remove_if(RandomAccessIterator first, RandomAccessIterator last, UnaryPredicate pred)
    {
        typedef typename std::iterator_traits<RandomAccessIterator>::difference_type diff_type;
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;

        const diff_type element_count = std::distance(first, last);
        if (element_count <= 0)
        {
            return first;
        }
        auto src_view = _details::create_section(first, element_count);

        concurrency::array<T> tmp(element_count);
        concurrency::array_view<T> tmp_view(tmp);

        //  Here copy_if() is used with the predicate inverted
        auto last_element =  amp_stl_algorithms::copy_if(first, last, begin(tmp_view), 
            [pred](const T& i) restrict(amp)
        {
            return static_cast<unsigned int>(!pred(i));
        });
        const int remaining_elements = static_cast<int>(std::distance(begin(tmp_view), last_element));
        if (remaining_elements > 0)
        {
            concurrency::copy(tmp_view.section(0, remaining_elements), src_view.section(0, remaining_elements));
        }
        return first + remaining_elements;
    }

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename T>
    RandomAccessIterator remove_copy( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        const T& value )
    {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        return amp_stl_algorithms::copy_if(first, last, dest_first, [=](T& v) restrict(amp) { return (v != value); });
    }

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator remove_copy_if( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        UnaryPredicate p )
    {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        return amp_stl_algorithms::copy_if(first, last, dest_first, [=](T& v) restrict(amp) { return !p(v); });
    }

    //----------------------------------------------------------------------------
    // replace, replace_if, replace_copy, replace_copy_if
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    void replace( RandomAccessIterator first, 
        RandomAccessIterator last,
        const T& old_value, 
        const T& new_value )
    {
        typedef std::iterator_traits<RandomAccessIterator>::value_type T;

        amp_stl_algorithms::replace_if(first, last, [=](const T& v) restrict(amp) { return (v == old_value); }, new_value);
    }

    template<typename RandomAccessIterator, typename UnaryPredicate, typename T>
    void replace_if( RandomAccessIterator first, 
        RandomAccessIterator last,
        UnaryPredicate p, 
        const T& new_value )
    {
        typedef std::iterator_traits<RandomAccessIterator>::difference_type difference_type;
        const int tile_size = 512;

        difference_type element_count = std::distance(first, last);
        if (element_count <= 0) 
        {
            return;
        }
        auto src_view = _details::create_section(first, element_count);

        concurrency::parallel_for_each(concurrency::tiled_extent<tile_size>(src_view.extent).pad(), 
            [element_count, new_value, src_view, p](concurrency::tiled_index<tile_size> tidx) restrict(amp)
        {
            int idx = tidx.global[0];
            if ((idx < element_count) && p(src_view[idx]))
            {
                src_view[idx] = new_value;
            }
        });
    }

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename T>
    RandomAccessIterator replace_copy( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        const T& old_value,
        const T& new_value )
    {
        return amp_stl_algorithms::replace_copy_if(first, last, dest_first, [=](const T& v) restrict(amp) { return (v == old_value); }, new_value);
    }

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename UnaryPredicate, typename T>
    RandomAccessIterator replace_copy_if( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        UnaryPredicate p,
        const T& new_value )
    {
        typedef std::iterator_traits<RandomAccessIterator>::difference_type difference_type;
        const int tile_size = 512;

        difference_type element_count = std::distance(first, last);
        if (element_count <= 0) 
        {
            return dest_first;
        }
        auto src_view = _details::create_section(first, element_count);
        auto dest_view = _details::create_section(dest_first, element_count);

        int last_changed_idx = 0;
        concurrency::array_view<int> last_changed_idx_av(1, &last_changed_idx);

        concurrency::parallel_for_each(concurrency::tiled_extent<tile_size>(src_view.extent).pad(), 
            [element_count, new_value, src_view, dest_view, last_changed_idx_av, p](concurrency::tiled_index<tile_size> tidx) restrict(amp)
        {
            int idx = tidx.global[0];
            if (idx < element_count)
            {
                if (p(src_view[idx]))
                {
                    dest_view[idx] = new_value;
                    concurrency::atomic_fetch_max(&last_changed_idx_av(0), idx + 1);
                }
                else
                {
                    dest_view[idx] = src_view[idx];
                }
            }
        });

        return dest_first + last_changed_idx_av[0];
    }

    //----------------------------------------------------------------------------
    // reverse, reverse_copy
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator>
    void reverse( RandomAccessIterator first, RandomAccessIterator last )
    {
        typedef typename std::iterator_traits<RandomAccessIterator>::difference_type diff_type;
        const int tile_size = 512;

        const diff_type element_count = std::distance(first, last);
        if (element_count <= 1) 
        {
            return;
        }
        auto src_view = _details::create_section(first, element_count);
        const int last_element = element_count - 1;

        concurrency::tiled_extent<tile_size> compute_domain = concurrency::extent<1>(element_count >> 1);
        concurrency::parallel_for_each(compute_domain.pad(), [=] (concurrency::tiled_index<tile_size> tidx) restrict(amp) 
        {
            const int idx = tidx.global[0];
            if (idx < element_count)
            {
                amp_stl_algorithms::swap(src_view[idx], src_view[last_element - idx]);
            }
        });
    }

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator reverse_copy( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first)
    {
        typedef typename std::iterator_traits<RandomAccessIterator>::difference_type diff_type;
        const int tile_size = 512;

        const diff_type element_count = std::distance(first, last);
        if (element_count <= 0) 
        {
            return dest_first;
        }
        auto src_view = _details::create_section(first, element_count);
        auto dest_view = _details::create_section(dest_first, element_count);
        const int last_element = element_count - 1;

        concurrency::tiled_extent<tile_size> compute_domain = concurrency::extent<1>(element_count);
        concurrency::parallel_for_each(compute_domain.pad(), [=] (concurrency::tiled_index<tile_size> tidx) restrict(amp) 
        {
            const int idx = tidx.global[0];
            if (idx < element_count)
            {
                dest_view[idx] = src_view[last_element - idx];
            }
        });

        return dest_first + element_count;
    }

    //----------------------------------------------------------------------------
    // rotate, rotate_copy
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator rotate_copy(ConstRandomAccessIterator first,
        ConstRandomAccessIterator middle,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first)
    {
        typedef typename std::iterator_traits<RandomAccessIterator>::difference_type diff_type;
        const diff_type element_count = std::distance(first, last);
        const diff_type middle_offset = std::distance(first, middle);

        auto src_view = _details::create_section(first, element_count);
        auto dest_view = _details::create_section(dest_first, element_count);

        concurrency::parallel_for_each(dest_view.extent, [=](concurrency::index<1> idx) restrict(amp)
        {
            dest_view[idx] = src_view[(idx + middle_offset) % element_count];
        });
        return dest_first + element_count;
    }

    //----------------------------------------------------------------------------
    // search, search_n, binary_search
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // set_difference, set_intersection, set_symetric_distance, set_union
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // shuffle, random_shuffle, 
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // is_sorted, is_sorted_until, sort, partial_sort, partial_sort_copy, stable_sort
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename Compare>
    ConstRandomAccessIterator is_sorted_until( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::difference_type diff_type;
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::value_type T;

        const diff_type element_count = std::distance(first, last);
        if (element_count <= 1)
        {
            return last;
        }
        return _details::adjacent_find(first, element_count, 
            [=](const T& a, const T& b) restrict(amp) { return !comp(a, b); }) + 1;
    }

    template<typename ConstRandomAccessIterator>
    bool is_sorted( ConstRandomAccessIterator first, ConstRandomAccessIterator last )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::value_type T;
        return (amp_stl_algorithms::is_sorted_until(first, last, amp_algorithms::less_equal<T>()) == last);
    }

    template<typename ConstRandomAccessIterator, typename Compare>
    bool is_sorted( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp )
    {
        return (amp_stl_algorithms::is_sorted_until(first, last, comp) == last);
    }

    template<typename ConstRandomAccessIterator>
    ConstRandomAccessIterator is_sorted_until( ConstRandomAccessIterator first, ConstRandomAccessIterator last )
    {
        typedef typename std::iterator_traits<ConstRandomAccessIterator>::value_type T;
        return amp_stl_algorithms::is_sorted_until(first, last, amp_algorithms::less_equal<T>());
    }

    //----------------------------------------------------------------------------
    // swap, swap<T, N>, swap_ranges, iter_swap
    //----------------------------------------------------------------------------

    template<typename T>
    void swap( T& a, T& b ) restrict(cpu, amp)
    {
        T tmp = a;
        a = b;
        b = tmp;
    }

    template<typename T, int N>
    void swap( T (&a)[N], T (&b)[N]) restrict(cpu, amp)
    {
        for (int i = 0; i < N; ++i)
        {
            amp_stl_algorithms::swap(a[i], b[i]);
        }
    }

    template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    RandomAccessIterator2 swap_ranges( RandomAccessIterator1 first1,
        RandomAccessIterator1 last1, 
        RandomAccessIterator2 first2 )
    {
        typedef std::iterator_traits<RandomAccessIterator1>::difference_type difference_type;
        typedef std::iterator_traits<RandomAccessIterator1>::value_type T;
        const int tile_size = 512;

        difference_type element_count = std::distance(first1, last1);
        if (element_count <= 0)
        {
            return first2;
        }
        concurrency::array_view<T> first1_view = _details::create_section(first1, element_count);
        concurrency::array_view<T> first2_view = _details::create_section(first2, element_count);

        concurrency::tiled_extent<tile_size> compute_domain = concurrency::extent<1>(element_count >> 1);
        concurrency::parallel_for_each(compute_domain.pad(), [=] (concurrency::tiled_index<tile_size> tidx) restrict(amp) 
        {
            const int idx = tidx.global[0];
            if (idx < element_count)
            {
                amp_stl_algorithms::swap(first1_view[idx], first2_view[idx]);
            }
        });

        return first2 + element_count;
    }

    template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    void iter_swap( RandomAccessIterator1 a, RandomAccessIterator2 b ) restrict(cpu, amp)
    {
        amp_stl_algorithms::swap(*a, *b);
    }

    //----------------------------------------------------------------------------
    // transform (Unary)
    //----------------------------------------------------------------------------

    // The "UnaryFunction" functor needs to be callable as "func(ConstRandomAccessIterator::value_type)".
    // The functor needs to be blittable and cannot contain any array, array_view, or textures.

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename UnaryFunction>
    RandomAccessIterator transform( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first,
        UnaryFunction func)
    {
        typedef std::iterator_traits<ConstRandomAccessIterator>::difference_type difference_type;

        difference_type element_count = std::distance(first, last);
        if (element_count <= 0)
        {
            return dest_first;
        }
        // TODO: input view should be const.
        auto input_view = _details::create_section(first, element_count);
        auto output_view = _details::create_section(dest_first, element_count);
        output_view.discard_data();

        concurrency::parallel_for_each(output_view.extent, [func,input_view,output_view] (concurrency::index<1> idx) restrict(amp) {
            output_view[idx] = func(input_view[idx]);
        });

        return dest_first;
    }

    // The "BinaryFunction" functor needs to be callable as "func(ConstRandomAccessIterator1::value_type, ConstRandomAccessIterator2::value_type)".
    // The functor needs to be blittable and cannot contain any array, array_view, or textures.

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename BinaryFunction>
    RandomAccessIterator transform( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        RandomAccessIterator dest_first,
        BinaryFunction func)
    {
        typedef std::iterator_traits<ConstRandomAccessIterator1>::difference_type difference_type;

        difference_type element_count = std::distance(first1, last1);
        if (element_count <= 0)
        {
            return dest_first;
        }
        // TODO: input view should be const.
        auto input1_view = _details::create_section(first1, element_count);
        auto input2_view = _details::create_section(first2, element_count);
        auto output_view = _details::create_section(dest_first, element_count);
        output_view.discard_data();

        concurrency::parallel_for_each(output_view.extent, [func,input1_view,input2_view,output_view] (concurrency::index<1> idx) restrict(amp) {
            output_view[idx] = func(input1_view[idx], input2_view[idx]);
        });

        return dest_first;
    }

    //----------------------------------------------------------------------------
    // unique, unique_copy
    //----------------------------------------------------------------------------

}// namespace amp_stl_algorithms
