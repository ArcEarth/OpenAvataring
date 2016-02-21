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
 * asum.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

//
// asum_helper
//   Functor for ASUM reduction 
//
    
template <typename ret_type, typename value_type, typename x_type, typename functor>
struct asum_helper
{
    asum_helper(const value_type& value, const functor& sum_op) restrict(cpu, amp)
        : init_value(value), op(sum_op) 
    {
    }

    // computes the sum of lhs and the absolute value of X[idx] and stores results in lhs
    void local_reduce(value_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        // this uses a non-standard absolute value calculation
        lhs += abs_1(X[concurrency::index<1>(idx)]);
    }

    // reduction of container vec
    ret_type global_reduce(const std::vector<value_type>& vec) const
    {
         return std::accumulate(vec.begin(), vec.end(), init_value);
    }

    value_type init_value;
    functor op;
};

} // namespace _detail

//-------------------------------------------------------------------------
// ASUM
//  computes the sum of the absolute values in a container.
//-------------------------------------------------------------------------

template <typename x_type>
typename real_type<typename x_type::value_type>::type asum(const concurrency::accelerator_view& av, const x_type& x)
{
    typedef typename x_type::value_type T;
    typedef typename real_type<T>::type real_type;

    // size
    const int n = x.extent[0];

    // tuning parameters
    static const unsigned int tile_size = 128;
    static const unsigned int max_tiles = 64;

    auto func = _detail::asum_helper<real_type, real_type, x_type, _detail::sum<real_type>>(real_type(), _detail::sum<real_type>());

    // call generic 1D reduction
    return _detail::reduce<tile_size, max_tiles, real_type, real_type>(av, n, x, func);
}

} // namespace ampblas
