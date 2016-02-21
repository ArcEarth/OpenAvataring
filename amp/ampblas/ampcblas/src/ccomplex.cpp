/* 
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
 * ampblas_ccomplex.cpp
 *
 * This file contains the implementation of user-defined type complex data type
 * for AMPBLAS C API.
 *
 *---------------------------------------------------------------------------*/
#include <assert.h>
#include "ampcblas_complex.h"

ampblas_fcomplex ampblas_fcomplex_new(const float real, const float imag)
{
    ampblas_fcomplex tmp = { real, imag };
    return tmp;
}

ampblas_dcomplex ampblas_dcomplex_new(const double real, const double imag)
{
    ampblas_dcomplex tmp = { real, imag };
    return tmp;
}

bool ampblas_fcomplex_equal(const ampblas_fcomplex* lhs, const ampblas_fcomplex* rhs)
{
    assert(lhs != nullptr && rhs != nullptr);
    return (lhs->real == rhs->real && lhs->imag == rhs->imag);
}

bool ampblas_dcomplex_equal(const ampblas_dcomplex* lhs, const ampblas_dcomplex* rhs)
{
    assert(lhs != nullptr && rhs != nullptr);
    return (lhs->real == rhs->real && lhs->imag == rhs->imag);
}
