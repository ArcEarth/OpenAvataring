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
 * symv.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/symv.h"

namespace ampcblas {

template <typename trans_op, typename value_type>
void symv(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, int n, value_type alpha, const value_type *a, int lda, const value_type* x, int incx, value_type beta, value_type *y, int incy)
{
    // recursive order adjustment
	if (order == AmpblasRowMajor)
    {
        symv<trans_op>(AmpblasColMajor, uplo == AmpblasLower ? AmpblasUpper : AmpblasLower,  n, alpha, a, lda, x, incx, beta, y, incy);
        return;
    }

	// quick return
	if (n == 0 || (alpha == value_type() && beta == value_type(1)))
		return;

	// argument check
	if (n < 0)
        argument_error("symv", 3);
    if (a == nullptr)
        argument_error("symv", 5);
    if (lda < n)
        argument_error("symv", 6);
    if (x == nullptr)
        argument_error("symv", 7);
    if (y == nullptr)
        argument_error("symv", 9);

	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);
    auto a_mat = make_matrix_view(n, n, a, lda);

	// call generic implementation
	ampblas::symv<trans_op>(get_current_accelerator_view(), cast(uplo), alpha, a_mat, x_vec, beta, y_vec);
}

} // namespace ampcblas

extern "C" {

void ampblas_ssymv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampcblas::symv<ampblas::_detail::noop>(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY) );
}

void ampblas_dsymv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampcblas::symv<ampblas::_detail::noop>(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY) );
}

void ampblas_chemv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *A, const int lda, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex* beta, ampblas_fcomplex *Y, const int incY)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    const fcomplex cbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::symv<ampblas::_detail::conjugate>(order, Uplo, N, calpha, ampblas_cast(A), lda, ampblas_cast(X), incX, cbeta, ampblas_cast(Y), incY) )
}

void ampblas_zhemv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *A, const int lda, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex* beta, ampblas_dcomplex *Y, const int incY)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    const dcomplex zbeta  = *ampblas_cast(beta);
    AMPBLAS_CHECKED_CALL( ampcblas::symv<ampblas::_detail::conjugate>(order, Uplo, N, zalpha, ampblas_cast(A), lda, ampblas_cast(X), incX, zbeta, ampblas_cast(Y), incY) )
}

} // extern "C"
