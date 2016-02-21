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
 * gemm.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/gemm.h"

namespace ampcblas {

template <typename value_type>
void gemm(enum AMPBLAS_ORDER order, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, int m, int n, int k, value_type alpha, const value_type *a, int lda, const value_type *b, int ldb, value_type beta, value_type *c, int ldc) 
{
	// recursive order adjustment 
	if (order == AmpblasRowMajor)
    {
        gemm(AmpblasColMajor, transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
        return;
    }

    // quick return
    if ((m == 0 || n == 0 || alpha == value_type() || k == 0) && beta == value_type(1))
        return;

    // derived parameters
    auto a_row = (transa == AmpblasNoTrans ? m : k);
	auto a_col = (transa == AmpblasNoTrans ? k : m);   
	auto b_row = (transb == AmpblasNoTrans ? k : n);
	auto b_col = (transb == AmpblasNoTrans ? n : k);

	// error check
	if (m < 0)		       
		argument_error("gemm", 4);
	if (n < 0)        
		argument_error("gemm", 5);
	if (k < 0)        
		argument_error("gemm", 6);
	if (a == nullptr) 
		argument_error("gemm", 8);
	if (lda < a_row)
		argument_error("gemm", 9);
	if (b == nullptr) 
		argument_error("gemm", 10);
	if (ldb < b_row) 
		argument_error("gemm", 11);
	if (c == nullptr) 
		argument_error("gemm", 13);
	if (ldc < m) 
		argument_error("gemm", 14);
  
    // create views
	auto a_mat = make_matrix_view(a_row, a_col, a, lda);
	auto b_mat = make_matrix_view(b_row, b_col, b, ldb);
	auto c_mat = make_matrix_view(m, n, c, ldc);
    
    // special cases
	if (alpha == value_type())
	{
		if (beta == value_type())
			ampblas::_detail::fill(get_current_accelerator_view(), c_mat.extent, value_type(), c_mat);
		else
			ampblas::_detail::scale(get_current_accelerator_view(), c_mat.extent, beta, c_mat);
		return;
	}

    // forward to ampblas
    ampblas::gemm(get_current_accelerator_view(), cast(transa), cast(transb), alpha, a_mat, b_mat, beta, c_mat);
}

} // namespace ampcblas

extern "C" {

void ampblas_sgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void ampblas_dgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void ampblas_cgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const ampblas_fcomplex *alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const ampblas_fcomplex *beta, ampblas_fcomplex *C, const int ldc)
{
    const ampcblas::fcomplex calpha = *ampcblas::ampblas_cast(alpha);
    const ampcblas::fcomplex cbeta  = *ampcblas::ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::gemm(Order, TransA, TransB, M, N, K, calpha, ampcblas::ampblas_cast(A), lda, ampcblas::ampblas_cast(B), ldb, cbeta, ampcblas::ampblas_cast(C), ldc) );
}

void ampblas_zgemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const ampblas_dcomplex *alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const ampblas_dcomplex *beta, ampblas_dcomplex *C, const int ldc)
{
    const ampcblas::dcomplex zalpha = *ampcblas::ampblas_cast(alpha);
    const ampcblas::dcomplex zbeta  = *ampcblas::ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::gemm(Order, TransA, TransB, M, N, K, zalpha, ampcblas::ampblas_cast(A), lda, ampcblas::ampblas_cast(B), ldb, zbeta, ampcblas::ampblas_cast(C), ldc) );
}

} // extern "C" 