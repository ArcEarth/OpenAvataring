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
 * nrm2.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {
namespace _detail {

//
// nrm2_helper
//   Functor for NRM2 reduction 
//
template<typename ret_type, typename value_type, typename x_type, typename functor>
struct nrm2_helper
{
    nrm2_helper(const value_type& value, const functor& sum_op) restrict(cpu, amp) 
        : init_value(value), op(sum_op) 
    {
    }

    // computes the euclidean norm of lhs and the absolute value of X[idx] and stores results in lhs
    void local_reduce(value_type& lhs, int idx, const x_type& X) const restrict(cpu, amp)
    {
        value_type temp = X[ concurrency::index<1>(idx) ];
        lhs += (temp * temp);
    }

    // returns the square of the summation of all values in a container
    ret_type global_reduce(const std::vector<value_type>& vec) const
    {
         return std::sqrt(std::accumulate(vec.begin(), vec.end(), init_value));
    }

    value_type init_value;
    functor op;
};

} // namespace _detail

//-------------------------------------------------------------------------
// NRM2
//   computes the euclidean norm of a 1D container
//-------------------------------------------------------------------------

template <typename x_type>
typename x_type::value_type nrm2(const concurrency::accelerator_view& av, const x_type& x)
{
    typedef typename x_type::value_type T;

    // size
    const int n = x.extent[0];

    // tuning sizes
    static const unsigned int tile_size = 128;
    static const unsigned int max_tiles = 64;

    auto func = _detail::nrm2_helper<T, T, x_type, _detail::sum<T>>(T(), _detail::sum<T>());

    // call generic 1D reduction
    return _detail::reduce<tile_size, max_tiles, T, T>(av, n, x, func);
}

} // namespace ampblas
