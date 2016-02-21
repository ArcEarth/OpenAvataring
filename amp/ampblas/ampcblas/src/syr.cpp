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
 * syr.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/syr.h"

namespace ampcblas {

template <typename trans_op, typename alpha_type, typename value_type>
void syr(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, int n, alpha_type alpha, const value_type *x, int incx, value_type *a, int lda)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor)
    {
        syr<trans_op>(AmpblasColMajor, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, n, alpha, x, incx, a, lda);
        return;
    }

    // quick return
    if (n == 0 || alpha == alpha_type())
        return;

    // argument check
    if (n < 0)
        argument_error("syr", 3);
    if (x == nullptr)
        argument_error("syr", 5);
    if (a == nullptr)
        argument_error("syr", 7);
    if (lda < n)
        argument_error("syr", 8);

    // create views
    auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);

    // call generic implementation
    // TODO: move into ampblas
    if (uplo == AmpblasUpper)
        ampblas::syr<ampblas::uplo::upper,trans_op>(get_current_accelerator_view(), alpha, x_vec, a_mat);
    else
        ampblas::syr<ampblas::uplo::lower,trans_op>(get_current_accelerator_view(), alpha, x_vec, a_mat);
}

} // ampcblas

extern "C" {

void ampblas_ssyr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const float alpha, const float *X, const int incX, float *A, const int lda)
{
    AMPBLAS_CHECKED_CALL( ampcblas::syr<ampblas::_detail::noop>(order, uplo, N, alpha, X, incX, A, lda) );
}

void ampblas_dsyr(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const double alpha, const double *X, const int incX, double *A, const int lda)
{
	AMPBLAS_CHECKED_CALL( ampcblas::syr<ampblas::_detail::noop>(order, uplo, N, alpha, X, incX, A, lda) );
}

void ampblas_cher(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const float alpha, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *A, const int lda)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::syr<ampblas::_detail::conjugate>(order, uplo, N, alpha, ampblas_cast(X), incX, ampblas_cast(A), lda) );
}
 
void ampblas_zher(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO uplo, const int N, const double alpha, const ampblas_dcomplex *X, const int incX, ampblas_dcomplex *A, const int lda)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::syr<ampblas::_detail::conjugate>(order, uplo, N, alpha, ampblas_cast(X), incX, ampblas_cast(A), lda) );
}

} // extern "C"