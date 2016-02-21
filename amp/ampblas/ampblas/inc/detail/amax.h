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
 * amax.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

//
// amax_helper
//   Functor for AMAX reduction 
//

template<typename ret_type, typename value_type, typename x_type, typename functor>
struct amax_helper
{
    amax_helper(const value_type& value, const functor& max_op) restrict(cpu, amp) 
        : init_value(value), op(max_op) 
    {
    }

    // gets the maximum of the absolute values of lhs and X[idx], and stores in lhs
    void local_reduce(value_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        lhs = _detail::max(lhs, value_type(idx+1, abs_1( X[concurrency::index<1>(idx)] )));
    }

    // finds the maximum in a container and returns its position
    value_type global_reduce(const std::vector<value_type>& vec) const
    {
         return *std::max_element(vec.begin(), vec.end());
    }

    value_type init_value;
    functor op;
};

} // namespace _detail

//-------------------------------------------------------------------------
// AMAX
//   Finds the index of element having maximum absolute value in a container
//   This current implementation only uses Fortran (1-based) indexing
//-------------------------------------------------------------------------

template <typename int_type, typename x_type>
int_type amax(const concurrency::accelerator_view& av, const x_type& x)
{
    typedef typename x_type::value_type value_type;
    typedef typename real_type<value_type>::type real_type;
    typedef typename _detail::indexed_type<int_type, real_type> indexed_real_type;

    // size
    const int n = x.extent[0];

    // static and const for view in parallel section 
    static const unsigned int tile_size = 64;
    static const unsigned int max_tiles = 64;

    indexed_real_type x0 = indexed_real_type(1, real_type());
    auto func = _detail::amax_helper<indexed_real_type, indexed_real_type, x_type, _detail::maximum<indexed_real_type>>(x0, _detail::maximum<indexed_real_type>());

    // call generic 1D reduction
    indexed_real_type max = _detail::reduce<tile_size, max_tiles, indexed_real_type, indexed_real_type>(av, n, x, func);

    // return index
    return max.idx;
}

} // namespace ampblas
