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
 * copy.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/copy.h"

namespace ampcblas {

// Generic COPY algorithm for AMPBLAS arrays of type T
template <typename value_type>
void copy(int n, const value_type *x, int incx, value_type *y, int incy)
{
	// quick return
	if (n <= 0)
		return;

    // check arguments
    if (x == nullptr)
		argument_error("copy", 2);
    if (y == nullptr)
        argument_error("copy", 4);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

	ampblas::copy(get_current_accelerator_view(), x_vec, y_vec);
}

} // ampcblas

extern "C" {

void ampblas_scopy(const int N, const float *X, const int incX, float *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampcblas::copy(N, X, incX, Y, incY) );
}

void ampblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY)
{
	AMPBLAS_CHECKED_CALL( ampcblas::copy(N, X, incX, Y, incY) );
}

void ampblas_ccopy(const int N, const ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampcblas::copy(N, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

void ampblas_zcopy(const int N, const ampblas_dcomplex *X, const int incX, ampblas_dcomplex *Y, const int incY)
{
	AMPBLAS_CHECKED_CALL( ampcblas::copy(N, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

} // extern "C"