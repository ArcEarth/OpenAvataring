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
 * axpy.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/axpy.h"

namespace ampcblas {

// Generic AXPY algorithm for AMPBLAS arrays of type T
template <typename value_type>
void axpy(int n, value_type alpha, const value_type *x, int incx, value_type *y, int incy)
{
	// quick return
	if (n <= 0 || alpha == value_type())
        return;

    // check arguments
    if (x == nullptr)
		argument_error("axpy", 3);
    if (y == nullptr)
		argument_error("axpy", 5);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

    ampblas::axpy(get_current_accelerator_view(), alpha, x_vec, y_vec); 
}

} // namespace ampcblas

extern "C" {

void ampblas_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampcblas::axpy(N, alpha, X, incX, Y, incY) );
}

void ampblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY)
{
	AMPBLAS_CHECKED_CALL( ampcblas::axpy(N, alpha, X, incX, Y, incY) );
}

void ampblas_caxpy(const int N, const ampblas_fcomplex *alpha, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY)
{
    const ampcblas::fcomplex calpha = *ampcblas::ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::axpy(N, calpha, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

void ampblas_zaxpy(const int N, const ampblas_dcomplex *alpha, const ampblas_dcomplex *X, const int incX, ampblas_dcomplex *Y, const int incY)
{
    const ampcblas::dcomplex zalpha = *ampcblas::ampblas_cast(alpha);
	AMPBLAS_CHECKED_CALL( ampcblas::axpy(N, zalpha, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

} // extern "C"