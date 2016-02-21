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
 * trmv.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/trmv.h"

namespace ampcblas {

template <typename value_type>
void trmv(enum AMPBLAS_ORDER order, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int n, const value_type *a, int lda, value_type *x, int incx)
{
    // recursive order adjustment
	if (order == AmpblasRowMajor)
    {
        trmv(AmpblasColMajor, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, transa == AmpblasNoTrans ? AmpblasTrans : transa, diag, n, a, lda, x, incx);
        return;
    }

	// quick return
	if (n == 0)
		return;

	// argument check
	if (n < 0)
        argument_error("trmv", 5);
    if (a == nullptr)
        argument_error("trmv", 6);
    if (lda < n)
        argument_error("trmv", 7);
    if (x == nullptr)
        argument_error("trmv", 7);

    // workspace to enable some more parallelism
    concurrency::array<value_type,1> workspace(n);
    
	// create views
	auto x_vec = make_vector_view(n, x, incx);
    auto a_mat = make_matrix_view(n, n, a, lda);
    concurrency::array_view<value_type,1> y_vec(workspace);

    // call generic implementation
	ampblas::trmv(get_current_accelerator_view(), cast(uplo), cast(transa), cast(diag), a_mat, x_vec, y_vec);
   
    // copy workspace back to x
    auto x_2d = x_vec.get_base_view().view_as(concurrency::extent<2>(n,std::abs(incx))).section(concurrency::extent<2>(n,1));
    auto y_2d = y_vec.view_as(concurrency::extent<2>(n,1));
    y_2d.copy_to(x_2d);
    x_vec.get_base_view().refresh();
}

} // namespace ampcblas

extern "C" {

void ampblas_strmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampcblas::trmv(order, Uplo, TransA, Diag, N, A, lda, X, incX) );
}

void ampblas_dtrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampcblas::trmv(order, Uplo, TransA, Diag, N, A, lda, X, incX) );
}

void ampblas_ctrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *X, const int incX)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::trmv(order, Uplo, TransA, Diag, N, ampblas_cast(A), lda, ampblas_cast(X), incX) );
}

void ampblas_ztrmv(const enum AMPBLAS_ORDER order, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int N, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *X, const int incX)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::trmv(order, Uplo, TransA, Diag, N, ampblas_cast(A), lda, ampblas_cast(X), incX) );
}

} // extern "C"
