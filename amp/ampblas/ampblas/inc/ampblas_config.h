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
 * ampblas_config.h
 *
 * Common header for all ampblas implementation files. Contains shared headers
 * and utilites used for the implementation of all kernels.
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_CONFIG_H
#define AMPBLAS_CONFIG_H

#include <numeric>
#include <algorithm>
#include <amp.h>

#include "ampblas_defs.h"
#include "ampblas_complex.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

// Main namespace holding all BLAS routines
#define AMPBLAS_NAMESPACE_BEGIN namespace ampblas {
#define AMPBLAS_NAMESPACE_END } // namespace ampblas

// The functions in the _detail namespace are used internally by the BLAS routine implementations
#define DETAIL_NAMESPACE_BEGIN namespace _detail {
#define DETAIL_NAMESPACE_END } // namespace _detail

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN


// array type-traits
template <typename T>
struct is_array_view
{
    static const bool value = false; 
};

template <typename value_type, unsigned int rank>
struct is_array_view<concurrency::array_view<value_type,rank>>
{
    static const bool value = true; 
};

template <typename value_type, unsigned int rank>
struct is_array_view<concurrency::array_view<const value_type,rank>>
{
   static const bool value = true; 
};

// extent helpers
inline concurrency::extent<1> make_extent(int n) restrict(cpu, amp)
{
	return concurrency::extent<1>(n);
}

inline concurrency::extent<2> make_extent(int m, int n) restrict(cpu, amp)
{
	return concurrency::extent<2>(n,m);
}

//
// indexed_type
//   A wrapper to represent the value and its position of an element in a container. 
//   One usage is to find the index of the maximum value in a container. 
//
template <typename index_type, typename value_type>
struct indexed_type
{
    indexed_type() restrict(cpu, amp)
        : idx(1), val(value_type())  {}

    indexed_type(index_type idx, const value_type& val) restrict(cpu, amp)
        : idx(idx), val(val)  {}

    bool operator>(const indexed_type<index_type,value_type>& rhs) const restrict(cpu, amp) 
    {
        return val > rhs.val; 
    }

    bool operator<(const indexed_type<index_type,value_type>& rhs) const restrict(cpu, amp) 
    {
        return val < rhs.val; 
    }

    bool operator==(const indexed_type<index_type,value_type>& rhs) const restrict(cpu, amp) 
    {
        return val == rhs.val; 
    }

    index_type idx;
    value_type val;
};

//
// bounds checked opeators
// 

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

template <bool enabled, typename value_type, typename operation>
inline void guarded_update(const concurrency::array_view<value_type,2>& A, const concurrency::index<2>& idx, const operation& op) restrict(cpu,amp)
{
    if (enabled && A.extent.contains(idx))
        op(A[idx]);
}

inline bool is_diag(const concurrency::index<2>& idx) restrict(cpu,amp)
{
    return idx[0] == idx[1];
}

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_CONFIG_H
