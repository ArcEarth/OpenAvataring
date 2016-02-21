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
 * symm.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/symm.h"

namespace ampcblas {

template <typename trans_op, typename value_type>
void symm(enum AMPBLAS_ORDER order, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, int m, int n, value_type alpha, const value_type* a, int lda, const value_type* b, int ldb, value_type beta, value_type* c, int ldc)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor)
    {
        symm<trans_op>(AmpblasColMajor, side == AmpblasLeft ? AmpblasRight : AmpblasLeft, uplo = AmpblasUpper ? AmpblasLower : AmpblasUpper, n, m, alpha, b, ldb, a, lda, beta, c, ldc);
        return;
    }
    
    // quick return
    if (m == 0 || n == 0 || (alpha == value_type() && beta == value_type(1)))
        return;

    // derived parameters
    int k = (side == AmpblasLeft ? m : n);

    // argument check
    if (m < 0)
        argument_error("symm", 4);
    if (n < 0)
        argument_error("symm", 5);
    if (a == nullptr)
        argument_error("symm", 7);
    if (lda < k)
        argument_error("symm", 8);
    if (b == nullptr)
        argument_error("symm", 9);
    if (ldb < m)
        argument_error("symm", 10);
    if (c == nullptr)
        argument_error("symm", 12);
    if (ldc < m)
        argument_error("symm", 13);

    // create views
    auto a_mat = make_matrix_view(k, k, a, lda);
    auto b_mat = make_matrix_view(m, n, b, ldb);
    auto c_mat = make_matrix_view(m, n, c, ldc);

    // use fill or scale if alpha is zero
    if (alpha == value_type())
    {
        if (beta == value_type())
            ampblas::_detail::fill(get_current_accelerator_view(), make_extent(m,n), value_type(), c_mat);
        else
            ampblas::_detail::scale(get_current_accelerator_view(), make_extent(m,n), beta, c_mat);
        return;
    }

    // forward to tuning routine
    ampblas::symm<trans_op>(get_current_accelerator_view(), cast(side), cast(uplo), m, n, alpha, a_mat, b_mat, beta, c_mat);
}

} // namespace ampcblas

extern "C" {

void ampblas_ssymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::symm<ampblas::_detail::noop>(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void ampblas_dsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::symm<ampblas::_detail::noop>(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void ampblas_csymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const ampblas_fcomplex* beta, ampblas_fcomplex *C, const int ldc)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    const fcomplex cbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::symm<ampblas::_detail::noop>(Order, Side, Uplo, M, N, calpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, cbeta, ampblas_cast(C), ldc) );
}

void ampblas_zsymm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const ampblas_dcomplex* beta, ampblas_dcomplex *C, const int ldc)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    const dcomplex zbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::symm<ampblas::_detail::noop>(Order, Side, Uplo, M, N, zalpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, zbeta, ampblas_cast(C), ldc) );
}

void ampblas_chemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const ampblas_fcomplex* beta, ampblas_fcomplex *C, const int ldc)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    const fcomplex cbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::symm<ampblas::_detail::conjugate>(Order, Side, Uplo, M, N, calpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, cbeta, ampblas_cast(C), ldc) );
}

void ampblas_zhemm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const int M, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const ampblas_dcomplex* beta, ampblas_dcomplex *C, const int ldc)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    const dcomplex zbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::symm<ampblas::_detail::conjugate>(Order, Side, Uplo, M, N, zalpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, zbeta, ampblas_cast(C), ldc) )
}

} // extern "C"
