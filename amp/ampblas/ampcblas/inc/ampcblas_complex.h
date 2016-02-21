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
 * ampblas_ccomplex.h
 *
 * This file contains the complex user-defined type and API for AMPBLAS C API.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPCBLAS_COMPLEX_H
#define AMPCBLAS_COMPLEX_H

#include "ampcblas_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ampblas_fcomplex
{
    float real;
    float imag;
};

struct ampblas_dcomplex
{
    double real;
    double imag;
};

AMPBLAS_DLL ampblas_fcomplex ampblas_fcomplex_new(const float real, const float imag);
AMPBLAS_DLL ampblas_dcomplex ampblas_dcomplex_new(const double real, const double imag);
AMPBLAS_DLL bool ampblas_fcomplex_equal(const ampblas_fcomplex* lhs, const ampblas_fcomplex* rhs);
AMPBLAS_DLL bool ampblas_dcomplex_equal(const ampblas_dcomplex* lhs, const ampblas_dcomplex* rhs);

#ifdef __cplusplus
}
#endif 
#endif // AMPCBLAS_COMPLEX_H
