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
 * geqrf.cpp
 *
 *---------------------------------------------------------------------------*/

#include <functional>

#include <amp.h>

#include "ampclapack.h"      
#include "amplapack_runtime.h"

#include "detail\geqrf.h"    

namespace _detail {

template <typename value_type>
amplapack_status do_geqrf(int m, int n, value_type* a, int lda, value_type* tau, int& info)
{
    // create interface functor
    // NOTE: VS11 std::bind doesn't support over 5 arguments; using a transport struct as a workaround
    std::function<void(concurrency::accelerator_view&)> f = std::bind(amplapack::_detail::geqrf_unpack<value_type>, std::placeholders::_1, amplapack::_detail::geqrf_params<value_type>(m, n, a, lda, tau));

    // execute using interface
    return amplapack::safe_call_interface(f, info);
}

} // namespace _detail

extern "C" {

amplapack_status amplapack_sgeqrf(int m, int n, float* a, int lda, float* tau, int* info)
{
    return _detail::do_geqrf(m, n, a, lda, tau, *info); 
}

amplapack_status amplapack_dgeqrf(int m, int n, double* a, int lda, double* tau, int* info)
{
    return _detail::do_geqrf(m, n, a, lda, tau, *info); 
}

amplapack_status amplapack_cgeqrf(int m, int n, amplapack_fcomplex* a, int lda, amplapack_fcomplex* tau, int* info)
{
    return _detail::do_geqrf(m, n, amplapack::amplapack_cast(a), lda, amplapack::amplapack_cast(tau), *info); 
}

amplapack_status amplapack_zgeqrf(int m, int n, amplapack_dcomplex* a, int lda, amplapack_dcomplex* tau, int* info)
{
    return _detail::do_geqrf(m, n, amplapack::amplapack_cast(a), lda, amplapack::amplapack_cast(tau), *info); 
}

} // extern "C"
