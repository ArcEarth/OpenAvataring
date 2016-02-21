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
 * LAPACK library header for AMP LAPACK.
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPCLAPACK_H
#define AMPCLAPACK_H

//----------------------------------------------------------------------------
// DLL export/import specifiers
//
// The export/import mechanism used here is the __declspec(export) method 
// supported by Microsoft Visual Studio, but any other export method supported
// by your development environment may be substituted.
//----------------------------------------------------------------------------

#ifndef AMPLAPACK_DLL 
#define AMPLAPACK_DLL __declspec(dllimport)
#else
#undef AMPLAPACK_DLL
#define AMPLAPACK_DLL __declspec(dllexport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

//----------------------------------------------------------------------------
// Complex Types
//---------------------------------------------------------------------------- 

struct amplapack_fcomplex
{
    float real;
    float imag;
};

struct amplapack_dcomplex
{
    double real;
    double imag;
};

//----------------------------------------------------------------------------
// Return Status Types
//----------------------------------------------------------------------------

enum amplapack_status
{
    amplapack_success,             // routine completed successfully
    amplapack_data_error,          // an error with the data prevented completion; see specific routine for details
    amplapack_argument_error,      // an argument was invalid; see info parameter for enumeration
    amplapack_memory_error,        // insuffecient memory on the accelerator to complete the operation
    amplapack_runtime_error,       // an error related to the C++ AMP runtime was encountered
    amplapack_internal_error,      // an unexpected error was encountered (bad index, out of bounds, etc)
    amplapack_unknown_error        // catch all 
};

//----------------------------------------------------------------------------
// LAPACK Routines
//---------------------------------------------------------------------------- 

AMPLAPACK_DLL amplapack_status amplapack_sgetrf(int m, int n, float* a, int lda, int* ipiv, int* info);
AMPLAPACK_DLL amplapack_status amplapack_dgetrf(int m, int n, double* a, int lda, int* ipiv, int* info);
AMPLAPACK_DLL amplapack_status amplapack_cgetrf(int m, int n, amplapack_fcomplex* a, int lda, int* ipiv, int* info);
AMPLAPACK_DLL amplapack_status amplapack_zgetrf(int m, int n, amplapack_dcomplex* a, int lda, int* ipiv, int* info);

AMPLAPACK_DLL amplapack_status amplapack_sgeqrf(int m, int n, float* a, int lda, float* tau, int* info);
AMPLAPACK_DLL amplapack_status amplapack_dgeqrf(int m, int n, double* a, int lda, double* tau, int* info);
AMPLAPACK_DLL amplapack_status amplapack_cgeqrf(int m, int n, amplapack_fcomplex* a, int lda, amplapack_fcomplex* tau, int* info);
AMPLAPACK_DLL amplapack_status amplapack_zgeqrf(int m, int n, amplapack_dcomplex* a, int lda, amplapack_dcomplex* tau, int* info);

AMPLAPACK_DLL amplapack_status amplapack_spotrf(char uplo, int n, float* a, int lda, int* info);
AMPLAPACK_DLL amplapack_status amplapack_dpotrf(char uplo, int n, double* a, int lda, int* info);
AMPLAPACK_DLL amplapack_status amplapack_cpotrf(char uplo, int n, amplapack_fcomplex* a, int lda, int* info);
AMPLAPACK_DLL amplapack_status amplapack_zpotrf(char uplo, int n, amplapack_dcomplex* a, int lda, int* info);

#ifdef __cplusplus
}
#endif

#endif // AMPCLAPACK_H
