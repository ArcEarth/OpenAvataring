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
 * syr2k.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/syr2k.h"

namespace ampcblas {

template <typename trans_op, typename beta_type, typename value_type>
void syr2k(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, value_type alpha, const value_type* a, int lda, const value_type* b, int ldb, beta_type beta, value_type* c, int ldc)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor) 
    {
        syr2k<trans_op>(AmpblasColMajor, uplo == AmpblasLower ? AmpblasUpper : AmpblasLower, trans == AmpblasNoTrans ? AmpblasTrans : trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    // quick return
    if (n == 0 || ((alpha == value_type() || k == 0) && beta == beta_type(1)))
        return;

    // derived parameters
    int nrowa = (trans == AmpblasNoTrans ? n : k);
    int ka = (trans == AmpblasNoTrans ? k : n);

    // argument check
    if (n < 0)
        argument_error("syr2k", 4);
    if (k < 0)
        argument_error("syr2k", 5);
    if (a == nullptr)
        argument_error("syr2k", 7);
    if (lda < nrowa)
        argument_error("syr2k", 8);
    if (a == nullptr)
        argument_error("syr2k", 9);
    if (ldb < nrowa)
        argument_error("syr2k", 10);
    if (c == nullptr)
        argument_error("syr2k", 12);
    if (ldc < n)
        argument_error("syr2k", 13);

    // create views
    auto a_mat = make_matrix_view(nrowa, ka, a, lda);
    auto b_mat = make_matrix_view(nrowa, ka, b, ldb);
    auto c_mat = make_matrix_view(n, n, c, ldc);

    // use triangular scale or fill if alpha is zero and beta is not 1
    if (alpha == value_type())
    {
        if (beta == beta_type())
            ampblas::_detail::fill(get_current_accelerator_view(), cast(uplo), value_type(), c_mat);
        else
            ampblas::_detail::scale(get_current_accelerator_view(), cast(uplo), beta, c_mat);
        return;
    }

    ampblas::syr2k<trans_op>(get_current_accelerator_view(), cast(uplo), cast(trans), n, k, alpha, a_mat, b_mat, beta, c_mat);
}

} // namespace ampcblas

extern "C" {

void ampblas_ssyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::syr2k<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void ampblas_dsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::syr2k<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void ampblas_csyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const ampblas_fcomplex* beta, ampblas_fcomplex *C, const int ldc)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    const fcomplex cbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::syr2k<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, calpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, cbeta, ampblas_cast(C), ldc) );
}

void ampblas_zsyr2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const ampblas_dcomplex* beta, ampblas_dcomplex *C, const int ldc)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    const dcomplex zbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::syr2k<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, zalpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, zbeta, ampblas_cast(C), ldc) );
}

void ampblas_cher2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *B, const int ldb, const float beta, ampblas_fcomplex *C, const int ldc)
{   
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::syr2k<ampblas::_detail::conjugate>(Order, Uplo, Trans, N, K, calpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, beta, ampblas_cast(C), ldc) );
}

void ampblas_zher2k(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *B, const int ldb, const double beta, ampblas_dcomplex *C, const int ldc)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::syr2k<ampblas::_detail::conjugate>(Order, Uplo, Trans, N, K, zalpha, ampblas_cast(A), lda, ampblas_cast(B), ldb, beta, ampblas_cast(C), ldc) );
}

} // extern "C"
