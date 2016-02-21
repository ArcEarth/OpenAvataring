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
* C++ AMP algorithms library.
*
* This file contains the C++ AMP algorithms
*---------------------------------------------------------------------------*/

// TODO: Does it really make a lot of sense to declare two namespaces or should everything be flattened into amp_algorithms?
// TODO: Here the functions are defined here. In the STL implementation they are defined in the main header file 
// and just declared in the public one. Is this by design?

#pragma once

#include <amp.h>

#include <xx_amp_algorithms_impl.h>
#include <xx_amp_stl_algorithms_impl_inl.h>
#include <amp_indexable_view.h>

namespace amp_algorithms
{
#pragma region Arithmetic, comparison, logical and bitwise operators

    //----------------------------------------------------------------------------
    // Bitwise operations
    //----------------------------------------------------------------------------

    template <typename T>
    class bit_and
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a & b);
        }
    };

    template <typename T>
    class bit_or
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a | b);
        }
    };

    template <typename T>
    class bit_xor
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a ^ b);
        }
    };

    namespace _details
    {
        static const unsigned int bit08 = 0x80;
        static const unsigned int bit16 = 0x8000;
        static const unsigned int bit32 = 0x80000000;

        template<unsigned int N, int MaxBit>
        struct is_bit_set
        {
            enum { value = (N & MaxBit) ? 1 : 0 };
        };
    };

    template<unsigned int N, unsigned int MaxBit>
    struct static_count_bits
    {
        enum { value = (_details::is_bit_set<N, MaxBit>::value + static_count_bits<N, (MaxBit >> 1)>::value) };
    };

    template<unsigned int N>
    struct static_count_bits < N, 0 >
    {
        enum { value = FALSE };
    };

    template <typename T>
    int count_bits(T value)
    {
        value = value - ((value >> 1) & 0x55555555);
        value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
        return (((value + (value >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }

    //----------------------------------------------------------------------------
    // Arithmetic operations
    //----------------------------------------------------------------------------

    template <typename T>
    class plus
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a + b);
        }
    };

    template <typename T>
    class minus
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a - b);
        }
    };

    template <typename T>
    class multiplies
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a * b);
        }
    };

    template <typename T>
    class divides
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a / b);
        }
    };

    template <typename T>
    class modulus
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a % b);
        }
    };

    template <typename T>
    class negates
    {
    public:
        T operator()(const T &a) const restrict(cpu, amp)
        {
            return (-a);
        }
    };

    template<int N, unsigned int P = 0>
    struct log2
    {
        enum { value = log2<N / 2, P + 1>::value };
    };

    template <unsigned int P>
    struct log2<0, P>
    {
        enum { value = P };
    };

    template <unsigned int P>
    struct log2<1, P>
    {
        enum { value = P };
    };

    template<unsigned int N>
    struct static_is_power_of_two
    {
        enum { value = ((static_count_bits<N, _details::bit32>::value == 1) ? TRUE : FALSE) };
    };

    // While 1 is technically 2^0, for the purposes of calculating 
    // tile size it isn't useful.

    template <>
    struct static_is_power_of_two<1>
    {
        enum { value = FALSE };
    };


    // TODO: Generalize this for other integer types.
    template <typename T>
    bool is_power_of_two(T value)
    {
        return count_bits(value) == 1;
    }


    //----------------------------------------------------------------------------
    // Comparison operations
    //----------------------------------------------------------------------------

    template <typename T>
    class equal_to
    {
    public:
        bool operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a == b);
        }
    };

    template <typename T>
    class not_equal_to
    {
    public:
        bool operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a != b);
        }
    };

    template <typename T>
    class greater
    {
    public:
        bool operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a > b);
        }
    };

    template <typename T>
    class less
    {
    public:
        bool operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a < b);
        }
    };

    template <typename T>
    class greater_equal
    {
    public:
        bool operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a >= b);
        }
    };

    template <typename T>
    class less_equal
    {
    public:
        bool operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return (a <= b);
        }
    };

#ifdef max
#error amp_algorithms encountered a definition of the macro max.
#endif

    template <typename T>
    class max
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return ((a < b) ? b : a);
        }
    };

#ifdef min
#error amp_algorithms encountered a definition of the macro min.
#endif

    template <typename T>
    class min
    {
    public:
        T operator()(const T &a, const T &b) const restrict(cpu, amp)
        {
            return ((a < b) ? a : b);
        }
    };

    //----------------------------------------------------------------------------
    // Logical operations
    //----------------------------------------------------------------------------

    template<class T>
    class logical_not
    {
    public:
        bool operator()(const T& a) const restrict(cpu, amp)
        {
            return (!a);
        }
    };

    template<class T>
    class logical_and
    {
    public:
        bool operator()(const T& a, const T& b) const restrict(cpu, amp)
        {
            return (a && b);
        }
    };

    template<class T>
    class logical_or
    {
    public:
        bool operator()(const T& a, const T& b) const restrict(cpu, amp)
        {
            return (a || b);
        }
    };

    // TODO_NOT_IMPLEMENTED: Implement not1() and not2() if appropriate.

#pragma endregion

#pragma region Byte pack and unpack, padded read and write

    //----------------------------------------------------------------------------
    // Byte pack and unpack
    //----------------------------------------------------------------------------

    template<int index, typename T>
    inline unsigned long pack_byte(const T& value) restrict(cpu, amp)
    {
        assert(value < 256);
        static_assert(index < sizeof(T), "Index out of range.");
        return (static_cast<unsigned long>(value) && 0xFF) << (index * CHAR_BIT);
    }

    template<typename T>
    inline unsigned long pack_byte(const T& value, const unsigned index) restrict(cpu, amp)
    {
        //assert(value < 256);
        //assert(index < sizeof(T));
        return (static_cast<unsigned long>(value) && 0xFF) << (index * CHAR_BIT);
    }

    template<int index, typename T>
    inline unsigned int unpack_byte(const T& value) restrict(cpu, amp)
    {
        static_assert(index < sizeof(T), "Index out of range.");
        return (value >> (index * CHAR_BIT)) & 0xFF;
    }

    template<typename T>
    inline unsigned int unpack_byte(const T& value, const unsigned index) restrict(cpu, amp)
    {
        //assert(index < sizeof(T));
        return (value >> (index * CHAR_BIT)) & 0xFF;
    }

    template<typename T>
    unsigned int bit_count() restrict(cpu, amp)
    {
        return sizeof(T) * CHAR_BIT;
    }

    //----------------------------------------------------------------------------
    // container padded_read & padded_write
    //----------------------------------------------------------------------------

    template <typename InputIndexableView, int N>
    inline typename InputIndexableView::value_type padded_read(const InputIndexableView& arr, const concurrency::index<N> idx) restrict(cpu, amp)
    {
        return arr.extent.contains(idx) ? arr[idx] : typename InputIndexableView::value_type();
    }

    template <typename InputIndexableView>
    inline typename InputIndexableView::value_type padded_read(const InputIndexableView& arr, const int idx) restrict(cpu, amp)
    {
        return padded_read<InputIndexableView, 1>(arr, concurrency::index<1>(idx));
    }

    template <typename InputIndexableView, int N>
    inline void padded_write(InputIndexableView& arr, const concurrency::index<N> idx, const typename InputIndexableView::value_type &value) restrict(cpu, amp)
    {
        if (arr.extent.contains(idx))
        {
            arr[idx] = value;
        }
    }

    template <typename InputIndexableView>
    inline void padded_write(InputIndexableView& arr, const int idx, const typename InputIndexableView::value_type &value) restrict(cpu, amp)
    {
        padded_write<InputIndexableView, 1>(arr, concurrency::index<1>(idx), value);
    }

    // TODO: Should this return an extent? Better name.
    template <int N, typename InputIndexableView>
    inline int tile_partial_data_size(const InputIndexableView& arr, tiled_index<N> tidx) restrict(amp)
    {
        return arr.extent.size() - tidx.tile[0] * tidx.tile_extent[0];
    }
    
#pragma endregion

    //----------------------------------------------------------------------------
    // fill
    //----------------------------------------------------------------------------

    template<typename OutputIndexableView, typename T>
    void fill(const concurrency::accelerator_view &accl_view, OutputIndexableView& output_view, const T& value)
    {
        ::amp_algorithms::generate(accl_view, output_view, [value]() restrict(amp) { return value; });
    }

    template<typename OutputIndexableView, typename T>
    void fill(OutputIndexableView& output_view, const T& value)
    {
        ::amp_algorithms::generate(output_view, [value]() restrict(amp) { return value; });
    }

    //----------------------------------------------------------------------------
    // generate
    //----------------------------------------------------------------------------

    template <typename OutputIndexableView, typename Generator>
    void generate(const concurrency::accelerator_view &accl_view, OutputIndexableView& output_view, const Generator& generator)
    {
        _details::parallel_for_each(accl_view, output_view.extent, [output_view, generator](concurrency::index<indexable_view_traits<OutputIndexableView>::rank> idx) restrict(amp) {
            output_view[idx] = generator();
        });
    }

    template <typename OutputIndexableView, typename Generator>
    void generate(OutputIndexableView& output_view, const Generator& generator)
    {
        ::amp_algorithms::generate(_details::auto_select_target(), output_view, generator);
    }

    //----------------------------------------------------------------------------
    // merge_sort
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: merge_sort
    template <typename T, typename BinaryOperator>
    void merge_sort(const concurrency::accelerator_view& accl_view, concurrency::array_view<unsigned int>& input_view, BinaryOperator op)
    {
    }

    template <typename T>
    void merge_sort(const concurrency::accelerator_view& accl_view, concurrency::array_view<unsigned int>& input_view)
    {
        ::amp_algorithms::merge_sort(accl_view, input_view, amp_algorithms::less<T>());
    }

    //----------------------------------------------------------------------------
    // radix_sort
    //----------------------------------------------------------------------------

    template <typename T>
    inline void radix_sort(const concurrency::accelerator_view& accl_view, concurrency::array_view<T>& input_view, concurrency::array_view<T>& output_view)
    {
        static const int bin_width = 2;
        static const int tile_size = 128;
        _details::radix_sort<T, tile_size, bin_width>(accl_view, input_view, output_view);
    }

    // TODO: input_view should be a const.
    template <typename T>
    inline void radix_sort(concurrency::array_view<T>& input_view, concurrency::array_view<T>& output_view)
    {
        radix_sort(_details::auto_select_target(), input_view, output_view);
    }

    template <typename T>
    inline void radix_sort(const concurrency::accelerator_view& accl_view, concurrency::array_view<T>& input_view)
    {
        radix_sort(accl_view, input_view, input_view);
    }

    template <typename T>
    inline void radix_sort(concurrency::array_view<T>& input_view)
    {
        radix_sort(_details::auto_select_target(), input_view, input_view);
    }

    //----------------------------------------------------------------------------
    // reduce
    //----------------------------------------------------------------------------

    // Generic reduction template for binary operators that are commutative and associative
    template <typename InputIndexableView, typename BinaryFunction>
    typename std::result_of<BinaryFunction(const typename indexable_view_traits<InputIndexableView>::value_type&, const typename indexable_view_traits<InputIndexableView>::value_type&)>::type
        reduce(const concurrency::accelerator_view &accl_view, const InputIndexableView &input_view, const BinaryFunction &binary_op)
    {
        const int tile_size = 512;
        return _details::reduce<tile_size, 10000, InputIndexableView, BinaryFunction>(accl_view, input_view, binary_op);
    }

    template <typename InputIndexableView, typename BinaryFunction>
    typename std::result_of<BinaryFunction(const typename indexable_view_traits<InputIndexableView>::value_type&, const typename indexable_view_traits<InputIndexableView>::value_type&)>::type
        reduce(const InputIndexableView &input_view, const BinaryFunction &binary_op)
    {
        return reduce(_details::auto_select_target(), input_view, binary_op);
    }

    // This header needs these enums but they also have to be defined in the _impl header for use by the 
    // main STL header, which includes the _impl.
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

    template <int TileSize, scan_mode _Mode, typename _BinaryFunc, typename InputIndexableView>
    inline void scan(const InputIndexableView& input_view, InputIndexableView& output_view, const _BinaryFunc& op)
    {
        _details::scan<TileSize, _Mode, _BinaryFunc>(_details::auto_select_target(), input_view, output_view, op);
    }

    template <typename IndexableView>
    void scan_exclusive(const concurrency::accelerator_view& accl_view, const IndexableView& input_view, IndexableView& output_view)
    {
        _details::scan<_details::scan_default_tile_size, amp_algorithms::scan_mode::exclusive>(accl_view, input_view, output_view, amp_algorithms::plus<IndexableView::value_type>());
    }

    template <typename IndexableView>
    void scan_exclusive(const IndexableView& input_view, IndexableView& output_view)
    {
        _details::scan<_details::scan_default_tile_size, amp_algorithms::scan_mode::exclusive>(_details::auto_select_target(), input_view, output_view, amp_algorithms::plus<typename IndexableView::value_type>());
    }

    template <typename IndexableView>
    void scan_inclusive(const concurrency::accelerator_view& accl_view, const IndexableView& input_view, IndexableView& output_view)
    {
        _details::scan<_details::scan_default_tile_size, amp_algorithms::scan_mode::inclusive>(accl_view, input_view, output_view, amp_algorithms::plus<IndexableView::value_type>());
    }

    template <typename IndexableView>
    void scan_inclusive(const IndexableView& input_view, IndexableView& output_view)
    {
        _details::scan<_details::scan_default_tile_size, amp_algorithms::scan_mode::inclusive>(_details::auto_select_target(), input_view, output_view, amp_algorithms::plus<typename IndexableView::value_type>());
    }

    //----------------------------------------------------------------------------
    // transform (unary)
    //----------------------------------------------------------------------------

    template <typename ConstInputIndexableView, typename OutputIndexableView, typename UnaryFunc>
    void transform(const concurrency::accelerator_view &accl_view, const ConstInputIndexableView& input_view, OutputIndexableView& output_view, const UnaryFunc& func)
    {
        _details::parallel_for_each(accl_view, output_view.extent, [input_view,output_view,func] (concurrency::index<indexable_view_traits<OutputIndexableView>::rank> idx) restrict(amp) {
            output_view[idx] = func(input_view[idx]);
        });
    }

    template <typename ConstInputIndexableView, typename OutputIndexableView, typename UnaryFunc>
    void transform(const ConstInputIndexableView& input_view, OutputIndexableView& output_view, const UnaryFunc& func)
    {
        ::amp_algorithms::transform(_details::auto_select_target(), input_view, output_view, func);
    }

    //----------------------------------------------------------------------------
    // transform (binary)
    //----------------------------------------------------------------------------

    template <typename ConstInputIndexableView1, typename ConstInputIndexableView2, typename OutputIndexableView, typename BinaryFunc>
    void transform(const concurrency::accelerator_view &accl_view, const ConstInputIndexableView1& input_view1, const ConstInputIndexableView2& input_view2, OutputIndexableView& output_view, const BinaryFunc& func)
    {
        _details::parallel_for_each(accl_view, output_view.extent, [input_view1,input_view2,output_view,func] (concurrency::index<indexable_view_traits<OutputIndexableView>::rank> idx) restrict(amp) {
            output_view[idx] = func(input_view1[idx], input_view2[idx]);
        });
    }

    template <typename ConstInputIndexableView1, typename ConstInputIndexableView2, typename OutputIndexableView, typename BinaryFunc>
    void transform(const ConstInputIndexableView1& input_view1, const ConstInputIndexableView2& input_view2, OutputIndexableView& output_view, const BinaryFunc& func)
    {
        ::amp_algorithms::transform(_details::auto_select_target(), input_view1, input_view2, output_view, func);
    }
} // namespace amp_algorithms
