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
 * Generic wrappers to BLAS levels 1,2,3 library header for AMP CBLAS.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPXBLAS_H
#define AMPXBLAS_H

#include "ampcblas.h"

//----------------------------------------------------------------------------
// Prototypes for level 1 BLAS routines
//----------------------------------------------------------------------------

// 
// Routines with standard 4 prefixes (s, d, c, z)
//
// TODO: add other routines

// ampblas_xaxpy
template<typename value_type> 
void ampblas_xaxpy(const int N, const value_type alpha, const value_type *X,
                   const int incX, value_type *Y, const int incY);

template<> 
inline void ampblas_xaxpy<float>(const int N, const float alpha, const float *X,
				                 const int incX, float *Y, const int incY)
{
	ampblas_saxpy(N, alpha, X, incX, Y, incY);
}

template<> 
inline void ampblas_xaxpy<double>(const int N, const double alpha, const double *X,
							      const int incX, double *Y, const int incY)
{
	ampblas_daxpy(N, alpha, X, incX, Y, incY);
}

template<> 
inline void ampblas_xaxpy<ampblas_fcomplex>(const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex *X,
							      const int incX, ampblas_fcomplex *Y, const int incY)
{
	ampblas_caxpy(N, &alpha, X, incX, Y, incY);
}

template<> 
inline void ampblas_xaxpy<ampblas_dcomplex>(const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex *X,
							      const int incX, ampblas_dcomplex *Y, const int incY)
{
	ampblas_zaxpy(N, &alpha, X, incX, Y, incY);
}

// ampblas_xcopy
template<typename value_type> 
void ampblas_xcopy(const int N, const value_type *X,
                   const int incX, value_type *Y, const int incY);

template<> 
inline void ampblas_xcopy<float>(const int N, const float *X,
				                 const int incX, float *Y, const int incY)
{
	ampblas_scopy(N, X, incX, Y, incY);
}

template<> 
inline void ampblas_xcopy<double>(const int N, const double *X,
							      const int incX, double *Y, const int incY)
{
	ampblas_dcopy(N, X, incX, Y, incY);
}

template<> 
inline void ampblas_xcopy<ampblas_fcomplex>(const int N, const ampblas_fcomplex *X,
							      const int incX, ampblas_fcomplex *Y, const int incY)
{
	ampblas_ccopy(N, X, incX, Y, incY);
}

template<> 
inline void ampblas_xcopy<ampblas_dcomplex>(const int N, const ampblas_dcomplex *X,
							      const int incX, ampblas_dcomplex *Y, const int incY)
{
	ampblas_zcopy(N, X, incX, Y, incY);
}

// ampblas_xscal
template<typename value_type, typename alpha_type> 
void ampblas_xscal(const int N, const alpha_type alpha, value_type *X, const int incX);

template<> 
inline void ampblas_xscal<float>(const int N, const float alpha, float *X, const int incX)
{
	ampblas_sscal(N, alpha, X, incX);
}

template<> 
inline void ampblas_xscal<double>(const int N, const double alpha, double *X, const int incX)
{
	ampblas_dscal(N, alpha, X, incX);
}

template<> 
inline void ampblas_xscal<ampblas_fcomplex>(const int N, const ampblas_fcomplex alpha, ampblas_fcomplex *X, const int incX)
{
	ampblas_cscal(N, &alpha, X, incX);
}

template<> 
inline void ampblas_xscal<ampblas_dcomplex>(const int N, const ampblas_dcomplex alpha, ampblas_dcomplex *X, const int incX)
{
	ampblas_zscal(N, &alpha, X, incX);
}

template<> 
inline void ampblas_xscal<ampblas_fcomplex>(const int N, const float alpha, ampblas_fcomplex *X, const int incX)
{
	ampblas_csscal(N, alpha, X, incX);
}

template<> 
inline void ampblas_xscal<ampblas_dcomplex>(const int N, const double alpha, ampblas_dcomplex *X, const int incX)
{
	ampblas_zdscal(N, alpha, X, incX);
}

// ampblas_xswap
template<typename value_type> 
void ampblas_xswap(const int N, value_type *X,
                   const int incX, value_type *Y, const int incY);

template<> 
inline void ampblas_xswap<float>(const int N, float *X,
				                 const int incX, float *Y, const int incY)
{
	ampblas_sswap(N, X, incX, Y, incY);
}

template<> 
inline void ampblas_xswap<double>(const int N, double *X,
							      const int incX, double *Y, const int incY)
{
	ampblas_dswap(N, X, incX, Y, incY);
}

template<> 
inline void ampblas_xswap<ampblas_fcomplex>(const int N, ampblas_fcomplex *X,
                                            const int incX, ampblas_fcomplex *Y, const int incY)
{
	ampblas_cswap(N, X, incX, Y, incY);
}

template<> 
inline void ampblas_xswap<ampblas_dcomplex>(const int N, ampblas_dcomplex *X,
                                            const int incX, ampblas_dcomplex *Y, const int incY)
{
	ampblas_zswap(N, X, incX, Y, incY);
}

// ampblas_xgemm
template<typename value_type> 
void ampblas_xgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                   const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const value_type alpha, const value_type *A,
                   const int lda, const value_type *B, const int ldb,
                   const value_type beta, value_type *C, const int ldc);

template<> 
inline void ampblas_xgemm<float>(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                                 const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                                 const int K, const float alpha, const float *A,
                                 const int lda, const float *B, const int ldb,
                                 const float beta, float *C, const int ldc)
{
	ampblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<> 
inline void ampblas_xgemm<double>(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                                  const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                                  const int K, const double alpha, const double *A,
                                  const int lda, const double *B, const int ldb,
                                  const double beta, double *C, const int ldc)
{
	ampblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<> 
inline void ampblas_xgemm<ampblas_fcomplex>(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                                            const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                                            const int K, const ampblas_fcomplex alpha, const ampblas_fcomplex *A,
                                            const int lda, const ampblas_fcomplex *B, const int ldb,
                                            const ampblas_fcomplex beta, ampblas_fcomplex *C, const int ldc)
{
	ampblas_cgemm(Order, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template<> 
inline void ampblas_xgemm<ampblas_dcomplex>(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA,
                                            const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N,
                                            const int K, const ampblas_dcomplex alpha, const ampblas_dcomplex *A,
                                            const int lda, const ampblas_dcomplex *B, const int ldb,
                                            const ampblas_dcomplex beta, ampblas_dcomplex *C, const int ldc)
{
	ampblas_zgemm(Order, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

#endif //AMPXBLAS_H
