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

#pragma once
#ifndef AMPXBLAS_H
#define AMPXBLAS_H

#include "ampcblas.h"

//----------------------------------------------------------------------------
// Templated Type Helpers
//----------------------------------------------------------------------------

template <typename value_type> struct real_type                   { typedef typename value_type type; };
template <>                    struct real_type<ampblas_fcomplex> { typedef               float type; };
template <>                    struct real_type<ampblas_dcomplex> { typedef              double type; };

//----------------------------------------------------------------------------
// Overloaded BLAS 1 Routines
//----------------------------------------------------------------------------

// ampblas_amax
template <typename value_type> int ampblas_ixamax( int N, const value_type*       X, int incX );
template <>             inline int ampblas_ixamax( int N, const float*            X, int incX ) { return ampblas_isamax( N, X, incX ); }
template <>             inline int ampblas_ixamax( int N, const double*           X, int incX ) { return ampblas_idamax( N, X, incX ); }
template <>             inline int ampblas_ixamax( int N, const ampblas_fcomplex* X, int incX ) { return ampblas_icamax( N, X, incX ); }
template <>             inline int ampblas_ixamax( int N, const ampblas_dcomplex* X, int incX ) { return ampblas_izamax( N, X, incX ); }

// ampblas_xasum
template <typename value_type> typename real_type<value_type>::type ampblas_xasum( int N, const value_type*       X, int incX );
template <>                                           inline  float ampblas_xasum( int N, const float*            X, int incX ) { return ampblas_sasum ( N, X, incX ); }
template <>                                           inline double ampblas_xasum( int N, const double*           X, int incX ) { return ampblas_dasum ( N, X, incX ); }
template <>                                           inline  float ampblas_xasum( int N, const ampblas_fcomplex* X, int incX ) { return ampblas_scasum( N, X, incX ); }
template <>                                           inline double ampblas_xasum( int N, const ampblas_dcomplex* X, int incX ) { return ampblas_dzasum( N, X, incX ); }

// ampblas_xaxpy
template <typename value_type> void ampblas_xaxpy(const int N, const value_type       alpha, const value_type       *X, const int incX, value_type       *Y, const int incY);
template <>             inline void ampblas_xaxpy(const int N, const float            alpha, const float            *X, const int incX, float            *Y, const int incY) { ampblas_saxpy(N,  alpha, X, incX, Y, incY); }
template <>             inline void ampblas_xaxpy(const int N, const double           alpha, const double           *X, const int incX, double           *Y, const int incY) { ampblas_daxpy(N,  alpha, X, incX, Y, incY); }
template <>             inline void ampblas_xaxpy(const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY) { ampblas_caxpy(N, &alpha, X, incX, Y, incY); }
template <>             inline void ampblas_xaxpy(const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex *X, const int incX, ampblas_dcomplex *Y, const int incY) { ampblas_zaxpy(N, &alpha, X, incX, Y, incY); }

// ampblas_xcopy
template <typename value_type> void ampblas_xcopy(const int N, const value_type       *X, const int incX, value_type       *Y, const int incY);
template <>             inline void ampblas_xcopy(const int N, const float            *X, const int incX, float            *Y, const int incY) { ampblas_scopy(N, X, incX, Y, incY); }
template <>             inline void ampblas_xcopy(const int N, const double           *X, const int incX, double           *Y, const int incY) { ampblas_dcopy(N, X, incX, Y, incY); } 
template <>             inline void ampblas_xcopy(const int N, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY) { ampblas_ccopy(N, X, incX, Y, incY); }
template <>             inline void ampblas_xcopy(const int N, const ampblas_dcomplex *X, const int incX, ampblas_dcomplex *Y, const int incY) { ampblas_zcopy(N, X, incX, Y, incY); }

// ampblas_xdot
template <typename value_type>           value_type ampblas_xdot(const int N, const value_type       *X, const int incX, const value_type       *Y, const int incY);
template <>                 inline            float ampblas_xdot(const int N, const float            *X, const int incX, const float            *Y, const int incY) { return ampblas_sdot(N, X, incX, Y, incY); }
template <>                 inline           double ampblas_xdot(const int N, const double           *X, const int incX, const double           *Y, const int incY) { return ampblas_ddot(N, X, incX, Y, incY); }
template <>                 inline ampblas_fcomplex ampblas_xdot(const int N, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY) { ampblas_fcomplex ret; ampblas_cdotc_sub(N, X, incX, Y, incY, &ret); return ret; }
template <>                 inline ampblas_dcomplex ampblas_xdot(const int N, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY) { ampblas_dcomplex ret; ampblas_zdotc_sub(N, X, incX, Y, incY, &ret); return ret; }

// ampblas_xnrm2
template <typename value_type> typename real_type<value_type>::type ampblas_xnrm2(const int N, const value_type       *X, const int incX);
template <>                                           inline  float ampblas_xnrm2(const int N, const float            *X, const int incX) { return ampblas_snrm2 (N, X, incX); }
template <>                                           inline double ampblas_xnrm2(const int N, const double           *X, const int incX) { return ampblas_dnrm2 (N, X, incX); }
template <>                                           inline  float ampblas_xnrm2(const int N, const ampblas_fcomplex *X, const int incX) { return ampblas_scnrm2(N, X, incX); }
template <>                                           inline double ampblas_xnrm2(const int N, const ampblas_dcomplex *X, const int incX) { return ampblas_dznrm2(N, X, incX); }

// ampblas_xrot
template <typename value_type>  void ampblas_xrot(const int N, value_type *X, int incX, value_type *Y, int incY, value_type c, value_type s);
template <>              inline void ampblas_xrot(const int N, float      *X, int incX, float      *Y, int incY, float      c, float      s) { ampblas_srot(N,X,incX,Y,incY,c,s); }
template <>              inline void ampblas_xrot(const int N, double     *X, int incX, double     *Y, int incY, double     c, double     s) { ampblas_drot(N,X,incX,Y,incY,c,s); }

// ampblas_xscal
template <typename value_type, typename alpha_type> void ampblas_xscal(const int N, const alpha_type       alpha, value_type       *X, const int incX);
template <>                                  inline void ampblas_xscal(const int N, const float            alpha, float            *X, const int incX) { ampblas_sscal (N,  alpha, X, incX); }
template <>                                  inline void ampblas_xscal(const int N, const double           alpha, double           *X, const int incX) { ampblas_dscal (N,  alpha, X, incX); } 
template <>                                  inline void ampblas_xscal(const int N, const ampblas_fcomplex alpha, ampblas_fcomplex *X, const int incX) { ampblas_cscal (N, &alpha, X, incX); }
template <>                                  inline void ampblas_xscal(const int N, const ampblas_dcomplex alpha, ampblas_dcomplex *X, const int incX) { ampblas_zscal (N, &alpha, X, incX); }
template <>                                  inline void ampblas_xscal(const int N, const float            alpha, ampblas_fcomplex *X, const int incX) { ampblas_csscal(N,  alpha, X, incX); }
template <>                                  inline void ampblas_xscal(const int N, const double           alpha, ampblas_dcomplex *X, const int incX) { ampblas_zdscal(N,  alpha, X, incX); }

// ampblas_xswap
template <typename value_type> void ampblas_xswap(const int N, value_type       *X, const int incX, value_type       *Y, const int incY);
template <>             inline void ampblas_xswap(const int N, float            *X, const int incX, float            *Y, const int incY) { ampblas_sswap(N, X, incX, Y, incY); }
template <>             inline void ampblas_xswap(const int N, double           *X, const int incX, double           *Y, const int incY) { ampblas_dswap(N, X, incX, Y, incY); }
template <>             inline void ampblas_xswap(const int N, ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY) { ampblas_cswap(N, X, incX, Y, incY); }
template <>             inline void ampblas_xswap(const int N, ampblas_dcomplex *X, const int incX, ampblas_dcomplex *Y, const int incY) { ampblas_zswap(N, X, incX, Y, incY); }

//----------------------------------------------------------------------------
// Overloaded BLAS 2 Routines
//----------------------------------------------------------------------------

// ampblas_xgemv
template <typename value_type> void ampblas_xgemv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N, const value_type       alpha, const value_type        *A, const int lda, const value_type       *X, const int incX, const value_type       beta, value_type       *Y, const int incY);
template <>             inline void ampblas_xgemv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N, const float            alpha, const float             *A, const int lda, const float            *X, const int incX, const float            beta, float            *Y, const int incY) { ampblas_sgemv(Order, TransA, M, N,  alpha, A, lda, X, incX,  beta, Y, incY); }
template <>             inline void ampblas_xgemv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N, const double           alpha, const double            *A, const int lda, const double           *X, const int incX, const double           beta, double           *Y, const int incY) { ampblas_dgemv(Order, TransA, M, N,  alpha, A, lda, X, incX,  beta, Y, incY); }
template <>             inline void ampblas_xgemv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex  *A, const int lda, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex beta, ampblas_fcomplex *Y, const int incY) { ampblas_cgemv(Order, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY); }
template <>             inline void ampblas_xgemv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const int M, const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex  *A, const int lda, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex beta, ampblas_dcomplex *Y, const int incY) { ampblas_zgemv(Order, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY); }

// ampblas_xger
template <typename value_type> void ampblas_xger( enum AMPBLAS_ORDER Order, int M, int N, value_type       alpha, const value_type*       X, int incX, const value_type*       Y, int incY, value_type*       A, int lda );
template <>             inline void ampblas_xger( enum AMPBLAS_ORDER Order, int M, int N, float            alpha, const float*            X, int incX, const float*            Y, int incY, float*            A, int lda ) { ampblas_sger ( Order, M, N,  alpha, X, incX, Y, incY, A, lda ); }
template <>             inline void ampblas_xger( enum AMPBLAS_ORDER Order, int M, int N, double           alpha, const double*           X, int incX, const double*           Y, int incY, double*           A, int lda ) { ampblas_dger ( Order, M, N,  alpha, X, incX, Y, incY, A, lda ); }
template <>             inline void ampblas_xger( enum AMPBLAS_ORDER Order, int M, int N, ampblas_fcomplex alpha, const ampblas_fcomplex* X, int incX, const ampblas_fcomplex* Y, int incY, ampblas_fcomplex* A, int lda ) { ampblas_cgerc( Order, M, N, &alpha, X, incX, Y, incY, A, lda ); }
template <>             inline void ampblas_xger( enum AMPBLAS_ORDER Order, int M, int N, ampblas_dcomplex alpha, const ampblas_dcomplex* X, int incX, const ampblas_dcomplex* Y, int incY, ampblas_dcomplex* A, int lda ) { ampblas_zgerc( Order, M, N, &alpha, X, incX, Y, incY, A, lda ); }

// ampblas_xsymv
template <typename value_type> void ampblas_xsymv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const int N, const value_type       alpha, const value_type       *A, const int lda, const value_type       *X, const int incX, const value_type       beta, value_type       *Y, const int incY);
template <>             inline void ampblas_xsymv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const int N, const float            alpha, const float            *A, const int lda, const float            *X, const int incX, const float            beta, float            *Y, const int incY) { ampblas_ssymv(Order, Uplo, N,  alpha, A, lda, X, incX,  beta, Y, incY); }
template <>             inline void ampblas_xsymv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const int N, const double           alpha, const double           *A, const int lda, const double           *X, const int incX, const double           beta, double           *Y, const int incY) { ampblas_dsymv(Order, Uplo, N,  alpha, A, lda, X, incX,  beta, Y, incY); }
template <>             inline void ampblas_xsymv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex beta, ampblas_fcomplex *Y, const int incY) { ampblas_chemv(Order, Uplo, N, &alpha, A, lda, X, incX, &beta, Y, incY); }
template <>             inline void ampblas_xsymv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex beta, ampblas_dcomplex *Y, const int incY) { ampblas_zhemv(Order, Uplo, N, &alpha, A, lda, X, incX, &beta, Y, incY); }

// ampblas_xsyr
template <typename value_type> void ampblas_xsyr( enum AMPBLAS_ORDER Order, enum AMPBLAS_UPLO UPLO, int N, typename real_type<value_type>::type alpha, const value_type*       X, int incX, value_type*       A, int lda );
template <>             inline void ampblas_xsyr( enum AMPBLAS_ORDER Order, enum AMPBLAS_UPLO UPLO, int N, float                                alpha, const float*            X, int incX, float*            A, int lda ) { ampblas_ssyr( Order, UPLO, N, alpha, X, incX, A, lda ); }
template <>             inline void ampblas_xsyr( enum AMPBLAS_ORDER Order, enum AMPBLAS_UPLO UPLO, int N, double                               alpha, const double*           X, int incX, double*           A, int lda ) { ampblas_dsyr( Order, UPLO, N, alpha, X, incX, A, lda ); }
template <>             inline void ampblas_xsyr( enum AMPBLAS_ORDER Order, enum AMPBLAS_UPLO UPLO, int N, float                                alpha, const ampblas_fcomplex* X, int incX, ampblas_fcomplex* A, int lda ) { ampblas_cher( Order, UPLO, N, alpha, X, incX, A, lda ); }
template <>             inline void ampblas_xsyr( enum AMPBLAS_ORDER Order, enum AMPBLAS_UPLO UPLO, int N, double                               alpha, const ampblas_dcomplex* X, int incX, ampblas_dcomplex* A, int lda ) { ampblas_zher( Order, UPLO, N, alpha, X, incX, A, lda ); }

// ampblas_xtrmv
template <typename value_type> void ampblas_xtrmv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const value_type       *A, const int lda, value_type       *X, const int incX);
template <>             inline void ampblas_xtrmv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const float            *A, const int lda, float            *X, const int incX) { ampblas_strmv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }
template <>             inline void ampblas_xtrmv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const double           *A, const int lda, double           *X, const int incX) { ampblas_dtrmv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }
template <>             inline void ampblas_xtrmv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *X, const int incX) { ampblas_ctrmv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }
template <>             inline void ampblas_xtrmv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *X, const int incX) { ampblas_ztrmv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }

// ampblas_xtrsv
template <typename value_type> void ampblas_xtrsv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const value_type       *A, const int lda, value_type       *X, const int incX);
template <>             inline void ampblas_xtrsv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const float            *A, const int lda, float            *X, const int incX) { ampblas_strsv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }
template <>             inline void ampblas_xtrsv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const double           *A, const int lda, double           *X, const int incX) { ampblas_dtrsv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }
template <>             inline void ampblas_xtrsv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *X, const int incX) { ampblas_ctrsv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }
template <>             inline void ampblas_xtrsv(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *X, const int incX) { ampblas_ztrsv(Order, Uplo, TransA, Diag, N, A, lda, X, incX); }

//----------------------------------------------------------------------------
// Overloaded BLAS 3 Routines
//----------------------------------------------------------------------------

// ampblas_xgemm
template <typename value_type> void ampblas_xgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const value_type       alpha, const value_type       *A, const int lda, const value_type       *B, const int ldb, const value_type       beta, value_type       *C, const int ldc);
template <>             inline void ampblas_xgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float            alpha, const float            *A, const int lda, const float            *B, const int ldb, const float            beta, float            *C, const int ldc) { ampblas_sgemm(Order, TransA, TransB, M, N, K,  alpha, A, lda, B, ldb,  beta, C, ldc); }
template <>             inline void ampblas_xgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double           alpha, const double           *A, const int lda, const double           *B, const int ldb, const double           beta, double           *C, const int ldc) { ampblas_dgemm(Order, TransA, TransB, M, N, K,  alpha, A, lda, B, ldb,  beta, C, ldc); }
template <>             inline void ampblas_xgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const ampblas_fcomplex alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const ampblas_fcomplex beta, ampblas_fcomplex *C, const int ldc) { ampblas_cgemm(Order, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc); }
template <>             inline void ampblas_xgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const ampblas_dcomplex alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const ampblas_dcomplex beta, ampblas_dcomplex *C, const int ldc) { ampblas_zgemm(Order, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc); }

// ampblas_xsymm
template <typename value_type> void ampblas_xsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const value_type       alpha, const value_type       *A, const int lda, const value_type       *B, const int ldb, const value_type       beta, value_type       *C, const int ldc);
template <>             inline void ampblas_xsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const float            alpha, const float            *A, const int lda, const float            *B, const int ldb, const float            beta, float            *C, const int ldc) { ampblas_ssymm(Order, Side, Uplo, M, N,  alpha, A, lda, B, ldb,  beta, C, ldc); }
template <>             inline void ampblas_xsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const double           alpha, const double           *A, const int lda, const double           *B, const int ldb, const double           beta, double           *C, const int ldc) { ampblas_dsymm(Order, Side, Uplo, M, N,  alpha, A, lda, B, ldb,  beta, C, ldc); }
template <>             inline void ampblas_xsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const ampblas_fcomplex beta, ampblas_fcomplex *C, const int ldc) { ampblas_chemm(Order, Side, Uplo, M, N, &alpha, A, lda, B, ldb, &beta, C, ldc); }
template <>             inline void ampblas_xsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const ampblas_dcomplex beta, ampblas_dcomplex *C, const int ldc) { ampblas_zhemm(Order, Side, Uplo, M, N, &alpha, A, lda, B, ldb, &beta, C, ldc); }

// ampblas_xsyrk
template <typename value_type> void ampblas_xsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const typename real_type<value_type>::type alpha, const value_type       *A, const int lda, const typename real_type<value_type>::type beta, value_type       *C, const int ldc);
template <>             inline void ampblas_xsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float                                alpha, const float            *A, const int lda, const float                                beta, float            *C, const int ldc) { ampblas_ssyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc); }
template <>             inline void ampblas_xsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double                               alpha, const double           *A, const int lda, const double                               beta, double           *C, const int ldc) { ampblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc); }
template <>             inline void ampblas_xsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float                                alpha, const ampblas_fcomplex *A, const int lda, const float                                beta, ampblas_fcomplex *C, const int ldc) { ampblas_cherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc); }
template <>             inline void ampblas_xsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double                               alpha, const ampblas_dcomplex *A, const int lda, const double                               beta, ampblas_dcomplex *C, const int ldc) { ampblas_zherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc); }

// ampblas_syrk2k
template <typename value_type> void ampblas_xsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const value_type       alpha, const value_type       *A, const int lda, const value_type       *B, const int ldb, const typename real_type<value_type>::type beta, value_type       *C, const int ldc);
template <>             inline void ampblas_xsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float            alpha, const float            *A, const int lda, const float            *B, const int ldb, const float                                beta, float            *C, const int ldc) { ampblas_ssyr2k(Order, Uplo, Trans, N, K,  alpha, A, lda, B, ldb, beta, C, ldc); }
template <>             inline void ampblas_xsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double           alpha, const double           *A, const int lda, const double           *B, const int ldb, const double                               beta, double           *C, const int ldc) { ampblas_dsyr2k(Order, Uplo, Trans, N, K,  alpha, A, lda, B, ldb, beta, C, ldc); }
template <>             inline void ampblas_xsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_fcomplex alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const float                                beta, ampblas_fcomplex *C, const int ldc) { ampblas_cher2k(Order, Uplo, Trans, N, K, &alpha, A, lda, B, ldb, beta, C, ldc); }
template <>             inline void ampblas_xsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_dcomplex alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const double                               beta, ampblas_dcomplex *C, const int ldc) { ampblas_zher2k(Order, Uplo, Trans, N, K, &alpha, A, lda, B, ldb, beta, C, ldc); }

// ampblas_trsm
template <typename value_type> void ampblas_xtrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const value_type       alpha, const value_type       *A, const int lda, value_type       *B, const int ldb);
template <>             inline void ampblas_xtrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const float            alpha, const float            *A, const int lda, float            *B, const int ldb) { ampblas_strsm(Order, Side, Uplo, TransA, Diag, M, N,  alpha, A, lda, B, ldb); }
template <>             inline void ampblas_xtrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const double           alpha, const double           *A, const int lda, double           *B, const int ldb) { ampblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N,  alpha, A, lda, B, ldb); }
template <>             inline void ampblas_xtrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *B, const int ldb) { ampblas_ctrsm(Order, Side, Uplo, TransA, Diag, M, N, &alpha, A, lda, B, ldb); }
template <>             inline void ampblas_xtrsm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *B, const int ldb) { ampblas_ztrsm(Order, Side, Uplo, TransA, Diag, M, N, &alpha, A, lda, B, ldb); }

// ampblas_trmm
template <typename value_type> void ampblas_xtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const value_type       alpha, const value_type       *A, const int lda, value_type       *B, const int ldb);
template <>             inline void ampblas_xtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const float            alpha, const float            *A, const int lda, float            *B, const int ldb) { ampblas_strmm(Order, Side, Uplo, TransA, Diag, M, N,  alpha, A, lda, B, ldb); }
template <>             inline void ampblas_xtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const double           alpha, const double           *A, const int lda, double           *B, const int ldb) { ampblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N,  alpha, A, lda, B, ldb); }
template <>             inline void ampblas_xtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_fcomplex alpha, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *B, const int ldb) { ampblas_ctrmm(Order, Side, Uplo, TransA, Diag, M, N, &alpha, A, lda, B, ldb); }
template <>             inline void ampblas_xtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_dcomplex alpha, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *B, const int ldb) { ampblas_ztrmm(Order, Side, Uplo, TransA, Diag, M, N, &alpha, A, lda, B, ldb); }

#endif //AMPXBLAS_H
