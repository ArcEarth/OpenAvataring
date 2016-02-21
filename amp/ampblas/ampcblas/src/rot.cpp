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
 * rot.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/rot.h"

namespace ampcblas {

template <typename value_type>
void rot(int n, value_type* x, int incx, value_type* y, int incy, value_type c, value_type s)
{
	// quick return
	if (n <= 0)
		return;

	// error check
	if (x == nullptr)
		argument_error("rot", 2);
	if (y == nullptr)
		argument_error("rot", 4);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

    ampblas::rot(get_current_accelerator_view(), x_vec, y_vec, c, s); 
}

} // namespace ampcblas

extern "C" {

void ampblas_srot(const int N, float *X, const int incX, float *Y, const int incY, const float c, const float s)
{
    AMPBLAS_CHECKED_CALL( ampcblas::rot(N, X, incX, Y, incY, c, s) );
}

void ampblas_drot(const int N, double *X, const int incX, double *Y, const int incY, const double c, const double s)
{    
    AMPBLAS_CHECKED_CALL( ampcblas::rot(N, X, incX, Y, incY, c, s) );
}

} // extern "C"
