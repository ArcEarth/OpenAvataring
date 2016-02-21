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
 * swap.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/swap.h"

namespace ampcblas {

template <typename value_type>
void swap(int n, value_type *x, int incx, value_type *y, int incy)
{
	// quick return
	if (n <= 0 || x == y) 
		return;
 
    // check arguments
    if (x == nullptr)
		argument_error("swap", 2);
	if (y == nullptr)
		argument_error("swap", 3);

    auto x_vec = make_vector_view(n,x,incx);
    auto y_vec = make_vector_view(n,y,incy);
    ampblas::swap(get_current_accelerator_view(), x_vec, y_vec);
}

} // namespace ampcblas

extern "C" {

void ampblas_sswap(const int N, float *X, const int incX, float *Y, const int incY)
{
    AMPBLAS_CHECKED_CALL( ampcblas::swap(N, X, incX, Y, incY) );
}

void ampblas_dswap(const int N, double *X, const int incX, double *Y, const int incY)
{
	AMPBLAS_CHECKED_CALL( ampcblas::swap(N, X, incX, Y, incY) );
}

void ampblas_cswap(const int N, ampblas_fcomplex *X, const int incX, ampblas_fcomplex *Y, const int incY)
{
    using ampcblas::ampblas_cast;

	AMPBLAS_CHECKED_CALL( ampcblas::swap(N, ampblas_cast(X), incX, ampblas_cast(Y), incY) );
}

void ampblas_zswap(const int N, ampblas_dcomplex *X, const int incX, ampblas_dcomplex *Y, const int incY)
{
    using ampcblas::ampblas_cast;

	AMPBLAS_CHECKED_CALL( ampcblas::swap(N, ampblas_cast(X), incX, ampblas_cast(Y), incY) );
}

} // extern "C"
