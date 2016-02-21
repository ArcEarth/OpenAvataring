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
 * ger.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/ger.h"

namespace ampcblas {

template <typename value_type, typename trans_op>
void ger(enum AMPBLAS_ORDER order, int m, int n, value_type alpha, const value_type *x, int incx, const value_type *y, int incy, value_type *a, int lda)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor)
    {
        ger<value_type,trans_op>(AmpblasColMajor, n, m, alpha, y, incy, x, incx, a, lda);
        return;
    }

    // quick return
    if (m == 0 || n == 0 || alpha == value_type())
        return;

    // argument check
    if (m < 0)
        argument_error("ger", 2);
    if (n < 0)
        argument_error("ger", 3);
    if (x == nullptr)
        argument_error("ger", 5);
    if (y == nullptr)
        argument_error("ger", 7);
    if (a == nullptr)
        argument_error("ger", 9);
    if (lda < (order == AmpblasColMajor ? m : n))
        argument_error("ger", 10);

    // create views
    auto x_vec = make_vector_view(m, x, incx);
    auto y_vec = make_vector_view(n, y, incy);
    auto a_mat = make_matrix_view(m, n, a, lda);

    // call generic implementation
    ampblas::ger<trans_op>(get_current_accelerator_view(), alpha, x_vec, y_vec, a_mat);
}

} // namespace ampcblas

extern "C" {

void ampblas_sger(const enum AMPBLAS_ORDER order, const int M, const int N, const float alpha, const float *X, const int incX, const float *Y, const int incY, float *A, const int lda)
{
    AMPBLAS_CHECKED_CALL( ampcblas::ger<float,ampblas::_detail::noop>(order, M, N, alpha, X, incX, Y, incY, A, lda) );
}

void ampblas_dger(const enum AMPBLAS_ORDER order, const int M, const int N, const double alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda)
{
	AMPBLAS_CHECKED_CALL( ampcblas::ger<double,ampblas::_detail::noop>(order, M, N, alpha, X, incX, Y, incY, A, lda) );
}

void ampblas_cgeru(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY, ampblas_fcomplex *A, const int lda)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::ger<fcomplex,ampblas::_detail::noop>(order, M, N, calpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
}

void ampblas_zgeru(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY, ampblas_dcomplex *A, const int lda)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
	AMPBLAS_CHECKED_CALL( ampcblas::ger<dcomplex,ampblas::_detail::noop>(order, M, N, zalpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
}

void ampblas_cgerc(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_fcomplex* alpha, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY, ampblas_fcomplex *A, const int lda)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::ger<fcomplex,ampblas::_detail::conjugate>(order, M, N, calpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
}

void ampblas_zgerc(const enum AMPBLAS_ORDER order, const int M, const int N, const ampblas_dcomplex* alpha, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY, ampblas_dcomplex *A, const int lda)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::ger<dcomplex,ampblas::_detail::conjugate>(order, M, N, zalpha, ampblas_cast(X), incX, ampblas_cast(Y), incY, ampblas_cast(A), lda) );
}

} // extern "C"
