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
 * ampblas_static.h
 *
 * Instantiates the implementations to be included in the static library
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_defs.h"
#include "ampblas_complex.h"
#include "ampblas_static.h"

#include "ampblas.h"

namespace ampblas {
namespace link {

//-----------------------------------------------------------------------------
// BLAS 3
//-----------------------------------------------------------------------------

void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, float           alpha, const concurrency::array_view<const float,2>&           a, const concurrency::array_view<const float,2>&           b, float           beta, const concurrency::array_view<float,2>&           c) { ampblas::gemm(av, transa, transb, alpha, a, b, beta, c); }
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, double          alpha, const concurrency::array_view<const double,2>&          a, const concurrency::array_view<const double,2>&          b, double          beta, const concurrency::array_view<double,2>&          c) { ampblas::gemm(av, transa, transb, alpha, a, b, beta, c); }
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, complex<float>  alpha, const concurrency::array_view<const complex<float>,2>&  a, const concurrency::array_view<const complex<float>,2>&  b, complex<float>  beta, const concurrency::array_view<complex<float>,2>&  c) { ampblas::gemm(av, transa, transb, alpha, a, b, beta, c); }
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, complex<double> alpha, const concurrency::array_view<const complex<double>,2>& a, const concurrency::array_view<const complex<double>,2>& b, complex<double> beta, const concurrency::array_view<complex<double>,2>& c) { ampblas::gemm(av, transa, transb, alpha, a, b, beta, c); }

void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, float           alpha, const concurrency::array_view<const float,2>&           a, const concurrency::array_view<float,2>&           b) { ampblas::trsm(av, side, uplo, transa, diag, alpha, a, b); }
void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, double          alpha, const concurrency::array_view<const double,2>&          a, const concurrency::array_view<double,2>&          b) { ampblas::trsm(av, side, uplo, transa, diag, alpha, a, b); }
void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, complex<float>  alpha, const concurrency::array_view<const complex<float>,2>&  a, const concurrency::array_view<complex<float>,2>&  b) { ampblas::trsm(av, side, uplo, transa, diag, alpha, a, b); }
void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, complex<double> alpha, const concurrency::array_view<const complex<double>,2>& a, const concurrency::array_view<complex<double>,2>& b) { ampblas::trsm(av, side, uplo, transa, diag, alpha, a, b); }

void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, float  alpha, const concurrency::array_view<const float,2>&           a_mat, float  beta, const concurrency::array_view<float,2>&           c_mat) { ampblas::herk(av, uplo, trans, alpha, a_mat, beta, c_mat); }
void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, double alpha, const concurrency::array_view<const double ,2>&         a_mat, double beta, const concurrency::array_view<double ,2>&         c_mat) { ampblas::herk(av, uplo, trans, alpha, a_mat, beta, c_mat); }
void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, float  alpha, const concurrency::array_view<const complex<float>,2>&  a_mat, float  beta, const concurrency::array_view<complex<float>,2>&  c_mat) { ampblas::herk(av, uplo, trans, alpha, a_mat, beta, c_mat); }
void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, double alpha, const concurrency::array_view<const complex<double>,2>& a_mat, double beta, const concurrency::array_view<complex<double>,2>& c_mat) { ampblas::herk(av, uplo, trans, alpha, a_mat, beta, c_mat); }

} // namespace link
} // namespace ampblas
