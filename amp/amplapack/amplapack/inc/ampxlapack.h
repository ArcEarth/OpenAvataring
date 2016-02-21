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
 * Overloaded C++ headers
 *
 *---------------------------------------------------------------------------*/

#pragma once
#ifndef AMPXLAPACK_H
#define AMPXLAPACK_H

#include "ampclapack.h"

//
// GETRF
//

inline amplapack_status amplapack_getrf(int m, int n, float* a, int lda, int* ipiv, int* info) 
{
    return amplapack_sgetrf(m, n, a, lda, ipiv, info);
}

inline amplapack_status amplapack_getrf(int m, int n, double* a, int lda, int* ipiv, int* info) 
{
    return amplapack_dgetrf(m, n, a, lda, ipiv, info);
}

inline amplapack_status amplapack_getrf(int m, int n, amplapack_fcomplex* a, int lda, int* ipiv, int* info) 
{
    return amplapack_cgetrf(m, n, a, lda, ipiv, info);
}

inline amplapack_status amplapack_getrf(int m, int n, amplapack_dcomplex* a, int lda, int* ipiv, int* info) 
{
    return amplapack_zgetrf(m, n, a, lda, ipiv, info);
}

//
// GEQRF
//

inline amplapack_status amplapack_geqrf(int m, int n, float* a, int lda, float* tau, int* info)
{
    return amplapack_sgeqrf(m, n, a, lda, tau, info);
}

inline amplapack_status amplapack_geqrf(int m, int n, double* a, int lda, double* tau, int* info)
{
    return amplapack_dgeqrf(m, n, a, lda, tau, info);
}

inline amplapack_status amplapack_geqrf(int m, int n, amplapack_fcomplex* a, int lda, amplapack_fcomplex* tau, int* info)
{
    return amplapack_cgeqrf(m, n, a, lda, tau, info);
}

inline amplapack_status amplapack_geqrf(int m, int n, amplapack_dcomplex* a, int lda, amplapack_dcomplex* tau, int* info)
{
    return amplapack_zgeqrf(m, n, a, lda, tau, info);
}

//
// POTRF
//

inline amplapack_status amplapack_potrf(char uplo, int n, float* a, int lda, int* info) 
{
    return amplapack_spotrf(uplo, n, a, lda, info);
}

inline amplapack_status amplapack_potrf(char uplo, int n, double* a, int lda, int* info) 
{
    return amplapack_dpotrf(uplo, n, a, lda, info);
}

inline amplapack_status amplapack_potrf(char uplo, int n, amplapack_fcomplex* a, int lda, int* info) 
{
    return amplapack_cpotrf(uplo, n, a, lda, info);
}

inline amplapack_status amplapack_potrf(char uplo, int n, amplapack_dcomplex* a, int lda, int* info) 
{
    return amplapack_zpotrf(uplo, n, a, lda, info);
}

#endif // AMPXLAPACK_H
