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
 * This header stamps out the function signatures of a few LAPACK functions
 * that should be located in the LAPACK host library. The host LAPACK library
 * can be specified by setting the LAPACK_LIB_PATH_[32/64] and 
 * LAPACK_LIB_FILES_[32/64] environment variables. If no host LAPACK library
 * is available it is possible to call non-hybrid routines that do not benefit
 * from overlapped execution.
 *
 *---------------------------------------------------------------------------*/

#ifndef LAPACK_HOST_H
#define LAPACK_HOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// use preprocessor options to match your LAPACK function signature
// -D_LAPACK_UPPER for DGETRF(...) (this is default)
// -D_LAPACK_UPPER_UNDERSCORE for DGETRF_(...)
// -D_LAPACK_LOWER for dgetrf(...)
// -D_LAPACK_LOWER_UNDERSCORE for dgetrf_(...)
#if defined(_LAPACK_UPPER)                       
#define LAPACK_NAME(lower, upper) upper
#elif defined(_LAPACK_UPPER_UNDERSCORE)
#define LAPACK_NAME(lower, upper) upper##_
#elif defined(_LAPACK_LOWER)
#define LAPACK_NAME(lower, upper) lower
#elif defined(_LAPACK_LOWER_UNDERSCORE)
#define LAPACK_NAME(lower, upper) lower##_
#else /* upper case default */          
#define LAPACK_NAME(lower, upper) upper
#endif

// long iteger support
// -D_LAPACK_ILP64 for 64-bit integers
#ifndef _LAPACK_ILP64
typedef int32_t lapack_int;
#else
typedef int64_t lapack_int;
#endif

// gemm name
#define LAPACK_SGEMM LAPACK_NAME(sgemm, SGEMM)
#define LAPACK_DGEMM LAPACK_NAME(dgemm, DGEMM)
#define LAPACK_CGEMM LAPACK_NAME(cgemm, CGEMM)
#define LAPACK_ZGEMM LAPACK_NAME(zgemm, ZGEMM)

// gemm signature
void LAPACK_SGEMM(const char*, const char*, const lapack_int*, const lapack_int*, const lapack_int*, const float*, const float*, const lapack_int*, const float*, const lapack_int*, const float*, float*, const lapack_int*);
void LAPACK_DGEMM(const char*, const char*, const lapack_int*, const lapack_int*, const lapack_int*, const double*, const double*, const lapack_int*, const double*, const lapack_int*, const double*, double*, const lapack_int*);
void LAPACK_CGEMM(const char*, const char*, const lapack_int*, const lapack_int*, const lapack_int*, const void*, const void*, const lapack_int*, const void*, const lapack_int*, const void*, void*, const lapack_int*);
void LAPACK_ZGEMM(const char*, const char*, const lapack_int*, const lapack_int*, const lapack_int*, const void*, const void*, const lapack_int*, const void*, const lapack_int*, const void*, void*, const lapack_int*);

// getrf name
#define LAPACK_SGETRF LAPACK_NAME(sgetrf, SGETRF)
#define LAPACK_DGETRF LAPACK_NAME(dgetrf, DGETRF)
#define LAPACK_CGETRF LAPACK_NAME(cgetrf, CGETRF)
#define LAPACK_ZGETRF LAPACK_NAME(zgetrf, ZGETRF)

// getrf signature
void LAPACK_SGETRF(lapack_int*, lapack_int*, float*, lapack_int*, lapack_int*, lapack_int*);
void LAPACK_DGETRF(lapack_int*, lapack_int*, double*, lapack_int*, lapack_int*, lapack_int*);
void LAPACK_CGETRF(lapack_int*, lapack_int*, void*, lapack_int*, lapack_int*, lapack_int*);
void LAPACK_ZGETRF(lapack_int*, lapack_int*, void*, lapack_int*, lapack_int*, lapack_int*);

// geqrf name
#define LAPACK_SGEQRF LAPACK_NAME(sgeqrf, SGEQRF)
#define LAPACK_DGEQRF LAPACK_NAME(dgeqrf, DGEQRF)
#define LAPACK_CGEQRF LAPACK_NAME(cgeqrf, CGEQRF)
#define LAPACK_ZGEQRF LAPACK_NAME(zgeqrf, ZGEQRF)

// geqrf signature
void LAPACK_SGEQRF(lapack_int*, lapack_int*, float*, lapack_int*, float*, float*, lapack_int*, lapack_int*);
void LAPACK_DGEQRF(lapack_int*, lapack_int*, double*, lapack_int*, double*, double*, lapack_int*, lapack_int*);
void LAPACK_CGEQRF(lapack_int*, lapack_int*, void*, lapack_int*, void*, void*, lapack_int*, lapack_int*);
void LAPACK_ZGEQRF(lapack_int*, lapack_int*, void*, lapack_int*, void*, void*, lapack_int*, lapack_int*);

// larft name
#define LAPACK_SLARFT LAPACK_NAME(slarft, SLARFT)
#define LAPACK_DLARFT LAPACK_NAME(dlarft, DLARFT)
#define LAPACK_CLARFT LAPACK_NAME(clarft, CLARFT)
#define LAPACK_ZLARFT LAPACK_NAME(zlarft, ZLARFT)

// larft signature
void LAPACK_SLARFT(char*, char*, lapack_int*, lapack_int*, float*, lapack_int*, float*, float*, lapack_int*);
void LAPACK_DLARFT(char*, char*, lapack_int*, lapack_int*, double*, lapack_int*, double*, double*, lapack_int*);
void LAPACK_CLARFT(char*, char*, lapack_int*, lapack_int*, void*, lapack_int*, void*, void*, lapack_int*);
void LAPACK_ZLARFT(char*, char*, lapack_int*, lapack_int*, void*, lapack_int*, void*, void*, lapack_int*);

// laswap name
#define LAPACK_SLASWP LAPACK_NAME(slaswp, SLASWP)
#define LAPACK_DLASWP LAPACK_NAME(dlaswp, DLASWP)
#define LAPACK_CLASWP LAPACK_NAME(claswp, CLASWP)
#define LAPACK_ZLASWP LAPACK_NAME(zlaswp, ZLASWP)

// laswp signature
void LAPACK_SLASWP(lapack_int*, float*, lapack_int*, lapack_int*, lapack_int*, lapack_int*, lapack_int*);
void LAPACK_DLASWP(lapack_int*, double*, lapack_int*, lapack_int*, lapack_int*, lapack_int*, lapack_int*);
void LAPACK_CLASWP(lapack_int*, void*, lapack_int*, lapack_int*, lapack_int*, lapack_int*, lapack_int*);
void LAPACK_ZLASWP(lapack_int*, void*, lapack_int*, lapack_int*, lapack_int*, lapack_int*, lapack_int*);

// orgqr/ungqr name
#define LAPACK_SORGQR LAPACK_NAME(sorgqr, SORGQR)
#define LAPACK_DORGQR LAPACK_NAME(dorgqr, DORGQR)
#define LAPACK_CUNGQR LAPACK_NAME(cungqr, CUNGQR)
#define LAPACK_ZUNGQR LAPACK_NAME(zungqr, ZUNGQR)

// orgqr/ungqr signature
void LAPACK_SORGQR(lapack_int*, lapack_int*, lapack_int*, float*, lapack_int*, float*, float*, lapack_int*, lapack_int*);
void LAPACK_DORGQR(lapack_int*, lapack_int*, lapack_int*, double*, lapack_int*, double*, double*, lapack_int*, lapack_int*);
void LAPACK_CUNGQR(lapack_int*, lapack_int*, lapack_int*, void*, lapack_int*, void*, void*, lapack_int*, lapack_int*);
void LAPACK_ZUNGQR(lapack_int*, lapack_int*, lapack_int*, void*, lapack_int*, void*, void*, lapack_int*, lapack_int*);

// potrf name
#define LAPACK_SPOTRF LAPACK_NAME(spotrf, SPOTRF)
#define LAPACK_DPOTRF LAPACK_NAME(dpotrf, DPOTRF)
#define LAPACK_CPOTRF LAPACK_NAME(cpotrf, CPOTRF)
#define LAPACK_ZPOTRF LAPACK_NAME(zpotrf, ZPOTRF)

// potrf signature
void LAPACK_SPOTRF(const char*, lapack_int*, float*, lapack_int*, lapack_int*);
void LAPACK_DPOTRF(const char*, lapack_int*, double*, lapack_int*, lapack_int*);
void LAPACK_CPOTRF(const char*, lapack_int*, void*, lapack_int*, lapack_int*);
void LAPACK_ZPOTRF(const char*, lapack_int*, void*, lapack_int*, lapack_int*);

#ifdef __cplusplus
}
#endif

#endif // LAPACK_HOST_H