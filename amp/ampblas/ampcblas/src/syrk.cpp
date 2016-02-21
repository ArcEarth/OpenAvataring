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
 * syrk.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/syrk.h"

namespace ampcblas {

template <typename trans_op, typename scalar_type, typename value_type>
void syrk(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, scalar_type alpha, const value_type* a, int lda, scalar_type beta, value_type* c, int ldc)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor)
    {
        syrk<trans_op>(AmpblasColMajor, uplo == AmpblasLower ? AmpblasUpper : AmpblasLower, trans == AmpblasNoTrans ? AmpblasTrans : trans, n, k, alpha, a, lda, beta, c, ldc);
        return;
    }

    // quick return
    if (n == 0 || ( (alpha == scalar_type() || k == 0) && beta == scalar_type(1)) )
        return;

    // derived prarameters
    int nrowa = (trans == AmpblasNoTrans ? n : k);
    int ka = (trans == AmpblasNoTrans ? k: n);

    // argument check
    if (n < 0)
        argument_error("syrk", 4);
    if (k < 0)
        argument_error("syrk", 5);
    if (a == nullptr)
        argument_error("syrk", 7);
    if (lda < nrowa)
        argument_error("syrk", 8);
    if (c == nullptr)
        argument_error("syrk", 10);
    if (ldc < n)
        argument_error("syrk", 11);

    // create views
    auto a_mat = make_matrix_view(nrowa, ka, a, lda);
    auto c_mat = make_matrix_view(n, n, c, ldc);

    // use triangular scale or fill if alpha is zero and beta is not 1
    if (alpha == scalar_type() && beta == scalar_type())
    {
        ampblas::_detail::fill(get_current_accelerator_view(), cast(uplo), value_type(), c_mat);
        return;
    }

    // forward to tuning routine
    ampblas::syrk<trans_op>(get_current_accelerator_view(), cast(uplo), cast(trans), alpha, a_mat, beta, c_mat);
}

} // namespace ampcblas

extern "C" {

void ampblas_ssyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const float *A, const int lda, const float beta, float *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::syrk<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc) );
}

void ampblas_dsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc)
{
    AMPBLAS_CHECKED_CALL( ampcblas::syrk<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc) );
}

void ampblas_csyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex* beta, ampblas_fcomplex *C, const int ldc)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    const fcomplex cbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::syrk<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, calpha, ampblas_cast(A), lda, cbeta, ampblas_cast(C), ldc) );
}

void ampblas_zsyrk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex* beta, ampblas_dcomplex *C, const int ldc)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    const dcomplex zbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::syrk<ampblas::_detail::noop>(Order, Uplo, Trans, N, K, zalpha, ampblas_cast(A), lda, zbeta, ampblas_cast(C), ldc) );
}

void ampblas_cherk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const float alpha, const ampblas_fcomplex *A, const int lda, const float beta, ampblas_fcomplex *C, const int ldc)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::syrk<ampblas::_detail::conjugate>(Order, Uplo, Trans, N, K, alpha, ampblas_cast(A), lda, beta, ampblas_cast(C), ldc) )
}

void ampblas_zherk(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const ampblas_dcomplex *A, const int lda, const double beta, ampblas_dcomplex *C, const int ldc)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::syrk<ampblas::_detail::conjugate>(Order, Uplo, Trans, N, K, alpha, ampblas_cast(A), lda, beta, ampblas_cast(C), ldc) )
}

} // extern "C"
