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
* This file contains the C++ AMP standard library algorithms
*---------------------------------------------------------------------------*/

#pragma once

#include <utility>
#include <xx_amp_algorithms_impl.h>
#include <xx_amp_algorithms_impl_inl.h>
#include <amp_iterators.h>

// TODO: Get the tests, header and internal implementations into the same logical order.
// TODO_NOT_IMPLEMENTED: consider supporting the heap functions (is_heap etc)

namespace amp_stl_algorithms
{
    //----------------------------------------------------------------------------
    // pair<T1, T2>
    //----------------------------------------------------------------------------

    template<class T1, class T2>
    class pair
    {
    public:
        typedef T1 first_type;
        typedef T2 second_type;

        T1 first;
        T2 second;

        pair() restrict(amp, cpu)
            : first(), second()
        { }

        pair(const T1& val1, const T2& val2) restrict(amp)
            : first(val1), second(val2)
        { }

        template<class Other1, class Other2>
        pair(const pair<Other1, Other2>& _Right) restrict(amp, cpu)
            : first(_Right.first), second(_Right.second)
        { }

        template<class Other1, class Other2>
        pair(Other1&& val1, Other2&& val2) restrict(amp, cpu)
            : first(val1), second(val2)
        { }

        // Support interop with std::pair.

        pair(const std::pair<T1, T2>& val) restrict(amp, cpu)
            : first(val.first), second(val.second)
        { }

        pair& operator=(const std::pair<T1, T2>& val) restrict(cpu)
        {
            first = val.first;
            second = val.second;
            return *this;
        }

        operator std::pair<T1, T2>() restrict(cpu)
        {
            return std::pair<T1, T2>(first, second);
        }
    };

    template<class T1, class T2>
    inline bool operator==(const pair<T1, T2>& _Left, const pair<T1, T2>& _Right) restrict(amp, cpu)
    {
        return (_Left.first == _Right.first && _Left.second == _Right.second);
    }

    template<class T1, class T2>
    inline bool operator!=(const pair<T1, T2>& _Left, const pair<T1, T2>& _Right) restrict(amp, cpu)
    {
        return (!(_Left == _Right));
    }

    template<class T1, class T2>
    inline bool operator<(const pair<T1, T2>& _Left, const pair<T1, T2>& _Right) restrict(amp, cpu)
    {
        return (_Left.first < _Right.first || (!(_Right.first < _Left.first) && _Left.second < _Right.second));
    }

    template<class T1, class T2>
    inline bool operator>(const pair<T1, T2>& _Left, const pair<T1, T2>& _Right) restrict(amp, cpu)
    {
        return (_Right < _Left);
    }

    template<class T1, class T2>
    inline bool operator<=(const pair<T1, T2>& _Left, const pair<T1, T2>& _Right) restrict(amp, cpu)
    {
        return (!(_Right < _Left));
    }

    template<class T1, class T2> 
    inline bool operator>=(const pair<T1, T2>& _Left, const pair<T1, T2>& _Right) restrict(amp, cpu)
    {
            return (!(_Left < _Right));
    }

    //----------------------------------------------------------------------------
    //  tuple<... T>
    //----------------------------------------------------------------------------

    // http://mitchnull.blogspot.com/2012/06/c11-tuple-implementation-details-part-1.html
    // TODO_NOT_IMPLEMENTED: Tuple<...T>
    /*
    template <typename... T>
    class tuple;

    namespace _details
    {

    }

    template <typename... T>
    class tuple {
    public:
        tuple();

        explicit tuple(const T&...);

        template <typename... U>
        explicit tuple(U&&...);

        tuple(const tuple&);

        tuple(tuple&&);

        template <typename... U>
        tuple(const tuple<U...>&);

        template <typename... U>
        tuple(tuple<U...>&&);

        tuple& operator=(const tuple&);
      
        tuple& operator=(tuple&&);

        template <typename... U>
        tuple& operator=(const tuple<U...>&);

        template <typename... U>
        tuple& operator=(tuple<U...>&&);

        void swap(tuple&);
    };

    template <typename... T> typename tuple_size<tuple<T...>>;

    template <size_t I, typename... T> typename tuple_element<I, tuple<T...>>;

    // element access:

    template <size_t I, typename... T>
    typename tuple_element<I, tuple<T...>>::type&
        get(tuple<T...>&);

    template <size_t I, typename... T>
    typename tuple_element<I, tuple<T...>>::type const&
        get(const tuple<T...>&);

    template <size_t I, typename... T>
    typename tuple_element<I, tuple<T...>>::type&&
        get(tuple<T...>&&);

    // relational operators:

    template<typename... T, typename... U>
    bool operator==(const tuple<T...>&, const tuple<U...>&);

    template<typename... T, typename... U>
    bool operator<(const tuple<T...>&, const tuple<U...>&);

    template<typename... T, typename... U>
    bool operator!=(const tuple<T...>&, const tuple<U...>&);

    template<typename... T, typename... U>
    bool operator>(const tuple<T...>&, const tuple<U...>&);

    template<typename... T, typename... U>
    bool operator<=(const tuple<T...>&, const tuple<U...>&);

    template<typename... T, typename... U>
    bool operator>=(const tuple<T...>&, const tuple<U...>&);

    template <typename... Types>
    void swap(tuple<Types...>& x, tuple<Types...>& y);
    */

    //----------------------------------------------------------------------------
    // adjacent_difference
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator>
    RandomAccessIterator adjacent_difference( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first ); 

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename BinaryOperation>
    RandomAccessIterator adjacent_difference( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first,
        BinaryOperation op );

    //----------------------------------------------------------------------------
    // all_of, any_of, none_of
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator,  typename UnaryPredicate>
    bool all_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p );

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    bool any_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p );

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    bool none_of( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p ); 

    // non-standard versions which store the result on the accelerator
    template<typename ConstRandomAccessIterator,  typename UnaryPredicate, typename OutputIterator>
    void all_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p, OutputIterator dest );

    template<typename ConstRandomAccessIterator,  typename UnaryPredicate, typename OutputIterator>
    void any_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p, OutputIterator dest );

    template<typename ConstRandomAccessIterator,  typename UnaryPredicate, typename OutputIterator>
    void none_of(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p, OutputIterator dest );

    //----------------------------------------------------------------------------
    // copy, copy_if, copy_n, copy_backward
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator copy( ConstRandomAccessIterator first,  ConstRandomAccessIterator last, RandomAccessIterator dest_beg );

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator copy_if( ConstRandomAccessIterator first,  
        ConstRandomAccessIterator last,
        RandomAccessIterator dest,
        UnaryPredicate p);

    template<typename ConstRandomAccessIterator, typename Size, typename RandomAccessIterator>
    RandomAccessIterator copy_n(ConstRandomAccessIterator first, Size count, RandomAccessIterator result);

    // TODO_NOT_IMPLEMENTED: copy_backward, does copy_backward really make any sense on a data-parallel context?
    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator copy_backward( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator d_last ); 

    //----------------------------------------------------------------------------
    // count, count_if
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename T>
    typename std::iterator_traits<ConstRandomAccessIterator>::difference_type
        count( ConstRandomAccessIterator first, ConstRandomAccessIterator last, const T &value ); 

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    typename std::iterator_traits<ConstRandomAccessIterator>::difference_type
        count_if( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p ); 

    //----------------------------------------------------------------------------
    // equal, equal_range
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    bool equal( ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1, ConstRandomAccessIterator2 first2 );

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename BinaryPredicate>
    bool equal( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1, 
        ConstRandomAccessIterator2 first2, 
        BinaryPredicate p );

    // TODO_NOT_IMPLEMENTED: equal_range
    template<typename ConstRandomAccessIterator, typename T>
    std::pair<ConstRandomAccessIterator, ConstRandomAccessIterator> 
        equal_range( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator, typename T, typename Compare>
    std::pair<ConstRandomAccessIterator,ConstRandomAccessIterator> 
        equal_range( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value, 
        Compare comp ); 

    //----------------------------------------------------------------------------
    // fill, fill_n
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    void fill( RandomAccessIterator first, RandomAccessIterator last, const T& value );

    // TODO: fill_n() should return an iterator for C++11 compliance.
    template<typename RandomAccessIterator, typename Size, typename T>
    void fill_n( RandomAccessIterator first, Size count, const T& value );

    // TODO_NOT_IMPLEMENTED: This fill_n differs only by return type. Probably better to implement the one that returns the end iterator than void.
    /*
    template<typename RandomAccessIterator, typename Size, typename T>
    RandomAccessIterator fill_n( RandomAccessIterator first, Size count, const T& value );*/

    //----------------------------------------------------------------------------
    // find, find_if, find_if_not, find_end, find_first_of, adjacent_find
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename T>
    ConstRandomAccessIterator find( ConstRandomAccessIterator first, ConstRandomAccessIterator last, const T& value );

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    ConstRandomAccessIterator find_if(ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p );

    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    ConstRandomAccessIterator find_if_not( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p );

    // TODO_NOT_IMPLEMENTED: find_end
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    ConstRandomAccessIterator1 find_end ( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2);

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename Predicate>
    ConstRandomAccessIterator1 find_end ( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2, 
        Predicate p);

    // TODO_NOT_IMPLEMENTED: find_first_of
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    ConstRandomAccessIterator1 find_first_of ( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2);

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename Predicate>
    ConstRandomAccessIterator1 find_first_of ( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2, 
        Predicate p);

    // TODO_NOT_IMPLEMENTED: adjacent_find
    template<typename ConstRandomAccessIterator>
    ConstRandomAccessIterator adjacent_find (ConstRandomAccessIterator first, ConstRandomAccessIterator last);

    template<typename ConstRandomAccessIterator, typename Predicate>
    ConstRandomAccessIterator adjacent_find (ConstRandomAccessIterator first,  ConstRandomAccessIterator last, Predicate p);

    //----------------------------------------------------------------------------
    // for_each, for_each_no_return
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator, typename UnaryFunction>
    UnaryFunction for_each( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryFunction f );

    // non-standard: no return
    template<typename ConstRandomAccessIterator, typename UnaryFunction>
    void for_each_no_return( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryFunction f );

    //----------------------------------------------------------------------------
    // generate, generate_n
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename Generator>
    void generate( RandomAccessIterator first, RandomAccessIterator last, Generator g );

    // TODO: generate_n() should return an iterator for C++11 compliance.
    template<typename RandomAccessIterator, typename Size, typename Generator>
    void generate_n( RandomAccessIterator first, Size count, Generator g );

    // TODO_NOT_IMPLEMENTED: This generate_n differs only by return type. Probably better to implement the one that returns the end iterator than void.
    /*
    template<typename RandomAccessIterator, typename Size, typename Generator>
    RandomAccessIterator generate_n( RandomAccessIterator first, Size count, Generator g );*/

    //----------------------------------------------------------------------------
    // includes
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    bool includes(ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, ConstRandomAccessIterator2 last2)
    {
        return includes(first1, last1, first2, last2, amp_algorithms::equal_to());
    }

    // TODO_NOT_IMPLEMENTED: includes
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename Compare>
    bool includes( ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, ConstRandomAccessIterator2 last2, 
        Compare comp );

    //----------------------------------------------------------------------------
    // inner_product
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename T>
    T inner_product(ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1, ConstRandomAccessIterator2 first2, const T value)
    {
        return amp_stl_algorithms::inner_product(first1, last1, first2, value, amp_algorithms::plus<T>(), amp_algorithms::multiplies<T>());
    }

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename T,
        typename BinaryOperation1, typename BinaryOperation2>
    T inner_product(ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1, ConstRandomAccessIterator2 first2, const T value,
        const BinaryOperation1& binary_op1, const BinaryOperation2& binary_op2);

    //----------------------------------------------------------------------------
    // iota
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    void iota( RandomAccessIterator first, RandomAccessIterator last, T value );

    //----------------------------------------------------------------------------
    // lexographical_compare
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    bool lexicographical_compare(ConstRandomAccessIterator1 first1,
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2,
        ConstRandomAccessIterator2 last2)
    {
        return amp_stl_algorithms::lexicographical_compare(first1, last1, first2, last2, amp_algorithms::less());
    }

    // TODO_NOT_IMPLEMENTED: lexicographical_compare
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename Compare>
    bool lexicographical_compare( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        Compare comp );

    // TODO: add decls for is_permuation, next_permutation, prev_permutation

    //----------------------------------------------------------------------------
    // lower_bound, upper_bound
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: lower_bound
    template<typename ConstRandomAccessIterator, typename T>
    ConstRandomAccessIterator lower_bound( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value ); 

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator, typename T, typename Compare>
    ConstRandomAccessIterator lower_bound( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value, Compare comp ); 

    // TODO_NOT_IMPLEMENTED: upper_bound
    template<typename ConstRandomAccessIterator, typename T>
    ConstRandomAccessIterator upper_bound( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator, typename T, typename Compare>
    ConstRandomAccessIterator upper_bound( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value, Compare comp ); 

    //----------------------------------------------------------------------------
    // merge, inplace_merge
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: merge
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator>
    RandomAccessIterator merge( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 fast1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2, 
        RandomAccessIterator dest_first);

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename BinaryPredicate>
    RandomAccessIterator merge( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2, 
        RandomAccessIterator dest_first,
        BinaryPredicate comp);

    // TODO_NOT_IMPLEMENTED: inplace_merge
    template<typename RandomAccessIterator>
    void inplace_merge( RandomAccessIterator first,
        RandomAccessIterator middle,
        RandomAccessIterator last ); 

    // NOT IMPLEMENTED
    template<typename RandomAccessIterator, typename Compare>
    void inplace_merge( RandomAccessIterator first,
        RandomAccessIterator middle,
        RandomAccessIterator last,
        Compare comp ); 

    //----------------------------------------------------------------------------
    // minmax, max_element, min_element, minmax_element
    //----------------------------------------------------------------------------

    template <typename T>
    inline amp_stl_algorithms::pair<const T, const T> minmax(const T a, const T b) restrict( cpu)
    {
        return minmax(a, b, amp_algorithms::less<T>());
    }

    template <typename T, typename Compare>
    inline amp_stl_algorithms::pair<const T, const T> minmax(const T a, const T b, Compare comp) restrict(cpu)
    {
        return comp(a, b) ? amp_stl_algorithms::pair<const T, const T>(a, b) : amp_stl_algorithms::pair<const T, const T>(b, a);
    }

    // TODO: enable initializer list in amp restricted code
    //
    // template<typename T>
    // T max( std::initializer_list<T> ilist) restrict(cpu,amp);
    //
    // template<typename T, typename Compare>
    // T max( std::initializer_list<T> ilist, Compare comp ); 
    //
    // template<typename T>
    // T min( std::initializer_list<T> ilist) restrict(cpu,amp);
    //
    // template<typename T, typename Compare>
    // T min( std::initializer_list<T> ilist, Compare comp ); 
    //
    // template<typename T>
    // std::pair<T,T> minmax( std::initializer_list ilist);
    //
    // template<typename T, typename Compare>
    // std::pair<T,T> minmax( std::initializer_list ilist, Compare comp ); 

    // TODO_NOT_IMPLEMENTED: max_element
    template<typename ConstRandomAccessIterator> 
    ConstRandomAccessIterator max_element( ConstRandomAccessIterator first, ConstRandomAccessIterator last );

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator, typename Compare>
    ConstRandomAccessIterator max_element( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp ); 

    // TODO_NOT_IMPLEMENTED: min_element
    template<typename ConstRandomAccessIterator> 
    ConstRandomAccessIterator min_element( ConstRandomAccessIterator first, ConstRandomAccessIterator last );

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator, typename Compare>
    ConstRandomAccessIterator min_element( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp ); 

    // TODO_NOT_IMPLEMENTED: minmax_element
    template<typename ConstRandomAccessIterator> 
    std::pair<ConstRandomAccessIterator,ConstRandomAccessIterator> 
        minmax_element( ConstRandomAccessIterator first, ConstRandomAccessIterator last ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator, typename Compare>
    std::pair<ConstRandomAccessIterator,ConstRandomAccessIterator> 
        minmax_element( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp ); 

    //----------------------------------------------------------------------------
    // mismatch
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: mismatch
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    std::pair<ConstRandomAccessIterator1, ConstRandomAccessIterator2>
        mismatch( ConstRandomAccessIterator1 first1, ConstRandomAccessIterator1 last1, ConstRandomAccessIterator2 first2 );

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename BinaryPredicate>
    std::pair<ConstRandomAccessIterator1,ConstRandomAccessIterator2>
        mismatch( ConstRandomAccessIterator1 first1,
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2,
        BinaryPredicate p );

    //----------------------------------------------------------------------------
    // move, move_backward
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: move
    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator move( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator d_first ); 

    // TODO_NOT_IMPLEMENTED: move_backward, does move_backward really make any sense on a data-parallel context?
    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator move_backward( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator d_last ); 

    //----------------------------------------------------------------------------
    // nth_element
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: nth_element
    template<typename RandomAccessIterator>
    void nth_element( RandomAccessIterator first, 
        RandomAccessIterator nth, 
        RandomAccessIterator last ); 

    // NOT IMPLEMENTED 
    template<typename RandomAccessIterator, typename Compare>
    void nth_element( RandomAccessIterator first, 
        RandomAccessIterator nth,
        RandomAccessIterator last, Compare comp ); 

    //----------------------------------------------------------------------------
    // partition, stable_partition, partition_point, is_partitioned
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: is_partitioned
    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    bool is_partitioned( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p );

    // TODO_NOT_IMPLEMENTED: partition
    template<typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator partition( RandomAccessIterator first, RandomAccessIterator last, UnaryPredicate comp);

    // TODO_NOT_IMPLEMENTED: stable_partition
    template<typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator stable_partition( RandomAccessIterator first, RandomAccessIterator last, UnaryPredicate p );

    // TODO_NOT_IMPLEMENTED: partition_point
    template<typename ConstRandomAccessIterator, typename UnaryPredicate>
    ConstRandomAccessIterator partition_point( ConstRandomAccessIterator first, ConstRandomAccessIterator last, UnaryPredicate p);

    //----------------------------------------------------------------------------
    // reduce
    //----------------------------------------------------------------------------

    // non-standard
    template<typename ConstRandomAccessIterator, typename T>
    T reduce( ConstRandomAccessIterator first, ConstRandomAccessIterator last, T init );

    // non-standard
    template<typename ConstRandomAccessIterator, typename T, typename BinaryOperation>
    T reduce( ConstRandomAccessIterator first, ConstRandomAccessIterator last, T init, BinaryOperation op ); 

    //----------------------------------------------------------------------------
    // remove, remove_if, remove_copy, remove_copy_if
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    RandomAccessIterator remove( RandomAccessIterator first, RandomAccessIterator last, const T& value );

    template<typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator remove_if( RandomAccessIterator first, RandomAccessIterator last, UnaryPredicate p );

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename T>
    RandomAccessIterator remove_copy( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        const T& value );

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename UnaryPredicate>
    RandomAccessIterator remove_copy_if( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        UnaryPredicate p );

    //----------------------------------------------------------------------------
    // replace, replace_if, replace_copy, replace_copy_if
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator, typename T>
    void replace( RandomAccessIterator first, 
        RandomAccessIterator last,
        const T& old_value, 
        const T& new_value );

    template<typename RandomAccessIterator, typename UnaryPredicate, typename T>
    void replace_if( RandomAccessIterator first, 
        RandomAccessIterator last,
        UnaryPredicate p, 
        const T& new_value ); 

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename T>
    RandomAccessIterator replace_copy( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        const T& old_value,
        const T& new_value );

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename UnaryPredicate, typename T>
    RandomAccessIterator replace_copy_if( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator dest_first,
        UnaryPredicate p,
        const T& new_value ); 

    //----------------------------------------------------------------------------
    // reverse, reverse_copy
    //----------------------------------------------------------------------------

    template<typename RandomAccessIterator>
    void reverse( RandomAccessIterator first, RandomAccessIterator last );

    template<typename ConstRandomAccessIterator, typename RandomAccessIterator>
    RandomAccessIterator reverse_copy( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first);

    //----------------------------------------------------------------------------
    // rotate, rotate_copy
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: rotate
    template<typename RandomAccessIterator>
    void rotate( RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last);

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator>
    RandomAccessIterator rotate_copy(ConstRandomAccessIterator first, 
        ConstRandomAccessIterator middle,
        ConstRandomAccessIterator last, 
        RandomAccessIterator dest_first);

    //----------------------------------------------------------------------------
    // search, search_n, binary_search
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: search
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2>
    ConstRandomAccessIterator1 search( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1, 
        ConstRandomAccessIterator2 first2,
        ConstRandomAccessIterator2 last2);

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2, typename Predicate>
    ConstRandomAccessIterator1 search( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2, 
        Predicate p);

    // TODO_NOT_IMPLEMENTED: search_n
    template<typename ConstRandomAccessIterator, typename Size, typename Type>
    ConstRandomAccessIterator search_n ( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        Size count, 
        const Type& val);

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator, typename Size, typename Type, typename Predicate>
    ConstRandomAccessIterator search_n ( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        Size count, 
        const Type& val, 
        Predicate p);

    // TODO_NOT_IMPLEMENTED: binary_search
    template<typename ConstRandomAccessIterator, typename T>
    bool binary_search( ConstRandomAccessIterator first, ConstRandomAccessIterator last, const T& value ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator, typename T, typename Compare>
    bool binary_search( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        const T& value, 
        Compare comp );

    //----------------------------------------------------------------------------
    // set_difference, set_intersection, set_symetric_distance, set_union
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: set_difference
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator>
    RandomAccessIterator set_difference( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename Compare>
    RandomAccessIterator set_difference( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first, 
        Compare comp ); 

    // TODO_NOT_IMPLEMENTED: set_intersection
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator>
    RandomAccessIterator set_intersection( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename Compare>
    RandomAccessIterator set_intersection( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first, 
        Compare comp ); 

    // TODO_NOT_IMPLEMENTED: set_symmetric_difference
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator>
    RandomAccessIterator set_symmetric_difference( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename Compare>
    RandomAccessIterator set_symmetric_difference( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first, 
        Compare comp); 

    // TODO_NOT_IMPLEMENTED: set_union
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator>
    RandomAccessIterator set_union( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first ); 

    // NOT IMPLEMENTED
    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename Compare>
    RandomAccessIterator set_union( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        ConstRandomAccessIterator2 last2,
        RandomAccessIterator d_first, 
        Compare comp ); 

    //----------------------------------------------------------------------------
    // shuffle, random_shuffle, 
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: random_shuffle
    template<typename RandomAccessIterator>
    void random_shuffle( RandomAccessIterator first, RandomAccessIterator last );

    // NOT IMPLEMENTED 
    template<typename RandomAccessIterator, typename RandomNumberGenerator>
    void random_shuffle( RandomAccessIterator first, 
        RandomAccessIterator last,
        RandomNumberGenerator& r );

    // NOT IMPLEMENTED 
    template<typename RandomAccessIterator, typename RandomNumberGenerator>
    void random_shuffle( RandomAccessIterator first, 
        RandomAccessIterator last, 
        RandomNumberGenerator&& r ); 

    // TODO_NOT_IMPLEMENTED: shuffle
    template<typename RandomAccessIterator, typename UniformRandomNumberGenerator>
    void shuffle( RandomAccessIterator first, 
        RandomAccessIterator last, 
        UniformRandomNumberGenerator&& g ); 

    //----------------------------------------------------------------------------
    // sort, partial_sort, partial_sort_copy, stable_sort, is_sorted, is_sorted_until
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator>
    bool is_sorted( ConstRandomAccessIterator first, ConstRandomAccessIterator last );

    template<typename ConstRandomAccessIterator, typename Compare>
    bool is_sorted( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp ); 

    template<typename ConstRandomAccessIterator>
    ConstRandomAccessIterator is_sorted_until( ConstRandomAccessIterator first, ConstRandomAccessIterator last );

    template<typename ConstRandomAccessIterator, typename Compare>
    ConstRandomAccessIterator is_sorted_until( ConstRandomAccessIterator first, ConstRandomAccessIterator last, Compare comp ); 

    // TODO_NOT_IMPLEMENTED: sort
    template<typename RandomAccessIterator>
    void sort( RandomAccessIterator first, RandomAccessIterator last );

    // NOT IMPLEMENTED 
    template<typename RandomAccessIterator, typename Compare>
    void sort( RandomAccessIterator first, RandomAccessIterator last, Compare comp ); 

    // TODO_NOT_IMPLEMENTED: partial_sort
    template<typename RandomAccessIterator>
    void partial_sort( RandomAccessIterator first, 
        RandomAccessIterator middle, 
        RandomAccessIterator last );

    // NOT IMPLEMENTED 
    template<typename RandomAccessIterator, typename Compare>
    void partial_sort( RandomAccessIterator first, 
        RandomAccessIterator middle,
        RandomAccessIterator last, Compare comp );

    // TODO_NOT_IMPLEMENTED: partial_sort_copy
    template<typename ConstRandomAccessIterator,typename RandomAccessIterator>
    RandomAccessIterator partial_sort_copy( ConstRandomAccessIterator first,
        ConstRandomAccessIterator last,
        RandomAccessIterator d_first, 
        RandomAccessIterator d_last ); 

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename Compare>
    RandomAccessIterator partial_sort_copy( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last,
        RandomAccessIterator d_first, 
        RandomAccessIterator d_last,
        Compare comp ); 

    // TODO_NOT_IMPLEMENTED: stable_sort
    template<typename RandomAccessIterator>
    void stable_sort( RandomAccessIterator first, RandomAccessIterator last );

    // NOT IMPLEMENTED 
    template<typename RandomAccessIterator, typename Compare>
    void stable_sort( RandomAccessIterator first, RandomAccessIterator last, Compare comp ); 

    //----------------------------------------------------------------------------
    // swap, swap_ranges, iter_swap
    //----------------------------------------------------------------------------

    template<typename T>
    void swap( T& a, T& b ) restrict(cpu, amp);

    template<typename T, int N>
    void swap( T (&a)[N], T (&b)[N]) restrict(cpu, amp);

    template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    RandomAccessIterator2 swap_ranges( RandomAccessIterator1 first1,
        RandomAccessIterator1 last1, 
        RandomAccessIterator2 first2 ) restrict(amp);

    template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    void iter_swap( RandomAccessIterator1 a, RandomAccessIterator2 b ) restrict(cpu, amp);

    //----------------------------------------------------------------------------
    // transform
    //----------------------------------------------------------------------------

    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename UnaryFunction>
    RandomAccessIterator transform( ConstRandomAccessIterator first1, 
        ConstRandomAccessIterator last1, 
        RandomAccessIterator result,
        UnaryFunction func);

    template<typename ConstRandomAccessIterator1, typename ConstRandomAccessIterator2,typename RandomAccessIterator, typename BinaryFunction>
    RandomAccessIterator transform( ConstRandomAccessIterator1 first1, 
        ConstRandomAccessIterator1 last1,
        ConstRandomAccessIterator2 first2, 
        RandomAccessIterator dest_first,
        BinaryFunction func);

    //----------------------------------------------------------------------------
    // unique, unique_copy
    //----------------------------------------------------------------------------

    // TODO_NOT_IMPLEMENTED: unique
    template<typename ConstRandomAccessIterator>
    ConstRandomAccessIterator unique( ConstRandomAccessIterator first, ConstRandomAccessIterator last);

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator, typename BinaryPredicate>
    ConstRandomAccessIterator unique( ConstRandomAccessIterator first, ConstRandomAccessIterator last, BinaryPredicate comp);

    // TODO_NOT_IMPLEMENTED: unique_copy
    template<typename ConstRandomAccessIterator,typename RandomAccessIterator>
    ConstRandomAccessIterator unique_copy( ConstRandomAccessIterator first, ConstRandomAccessIterator last, RandomAccessIterator d_first ); 

    // NOT IMPLEMENTED 
    template<typename ConstRandomAccessIterator,typename RandomAccessIterator, typename BinaryPredicate>
    ConstRandomAccessIterator unique_copy( ConstRandomAccessIterator first, 
        ConstRandomAccessIterator last, 
        RandomAccessIterator d_first, 
        BinaryPredicate p); 
}// namespace amp_stl_algorithms
