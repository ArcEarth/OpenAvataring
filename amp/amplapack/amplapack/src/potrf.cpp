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
 * potrf.cpp
 *
 *---------------------------------------------------------------------------*/

#include <functional>

#include <amp.h>

#include "ampclapack.h"      
#include "amplapack_runtime.h"

#include "detail\potrf.h"    

namespace _detail {

template <typename float_type>
amplapack_status do_potrf(char uplo, int n, float_type* a, int lda, int& info)
{
    // create interface functor
    std::function<void(concurrency::accelerator_view&)> f = std::bind(amplapack::potrf<float_type>, std::placeholders::_1, uplo, n, a, lda);

    // execute using interface
    return amplapack::safe_call_interface(f, info);
}

} // namespace _detail

extern "C" {

amplapack_status amplapack_spotrf(char uplo, int n, float* a, int lda, int* info)
{
    return _detail::do_potrf(uplo, n, a, lda, *info); 
}

amplapack_status amplapack_dpotrf(char uplo, int n, double* a, int lda, int* info)
{
    return _detail::do_potrf(uplo, n, a, lda, *info); 
}

amplapack_status amplapack_cpotrf(char uplo, int n, amplapack_fcomplex* a, int lda, int* info)
{
    return _detail::do_potrf(uplo, n, amplapack::amplapack_cast(a), lda, *info); 
}

amplapack_status amplapack_zpotrf(char uplo, int n, amplapack_dcomplex* a, int lda, int* info)
{
    return _detail::do_potrf(uplo, n, amplapack::amplapack_cast(a), lda, *info); 
}

} // extern "C"
