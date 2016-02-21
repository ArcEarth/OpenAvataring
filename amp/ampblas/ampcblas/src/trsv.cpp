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
 * trsv.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/trsv.h"

namespace ampcblas {

template <typename value_type>
void trsv(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int n, const value_type* a, int lda, value_type* x, int incx)
{
    // recursive order adjustment
	if (order == AmpblasRowMajor)
    {
        trsv(AmpblasColMajor, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, transa == AmpblasNoTrans ? AmpblasTrans : transa, diag, n, a, lda, x, incx);
        return;
    }

	// quick return
	if (n == 0)
		return;

	// argument check
	if (n < 0)
        argument_error("trsv", 5);
    if (a == nullptr)
        argument_error("trsv", 6);
    if (lda < n)
        argument_error("trsv", 7);
    if (x == nullptr)
        argument_error("trsv", 7);

	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);

    // forward to tuning routine
    ampblas::trsv(get_current_accelerator_view(), cast(uplo), cast(transa), cast(diag), a_mat, x_vec);
}

} // namespace ampcblas

extern "C" {

void ampblas_strsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampcblas::trsv(order, Uplo, TransA, Diag, N, A, lda, X, incX) );
}

void ampblas_dtrsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampcblas::trsv(order, Uplo, TransA, Diag, N, A, lda, X, incX) );
}

void ampblas_ctrsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *X, const int incX)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::trsv(order, Uplo, TransA, Diag, N, ampblas_cast(A), lda, ampblas_cast(X), incX) );
}

void ampblas_ztrsv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *X, const int incX)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::trsv(order, Uplo, TransA, Diag, N, ampblas_cast(A), lda, ampblas_cast(X), incX) );
}

} // extern "C"