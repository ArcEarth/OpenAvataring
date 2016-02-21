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
 * scal.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/scal.h"

namespace ampcblas {

// Generic SCAL algorithm for AMPBLAS arrays of type value_type
template <typename value_type, typename scalar_type>
void scal(int n, scalar_type alpha, value_type *x, int incx)
{
	// quick return
	if (n <= 0) 
        return;

    // check arguments
    if (x == nullptr)
		argument_error("scal", 3);

    auto x_vec = make_vector_view(n,x,incx);

    ampblas::scal(get_current_accelerator_view(), alpha, x_vec);
}

} // namespace ampcblas

extern "C" {

void ampblas_sscal(const int N, const float alpha, float *X, const int incX)
{
    AMPBLAS_CHECKED_CALL( ampcblas::scal(N, alpha, X, incX) );
}

void ampblas_dscal(const int N, const double alpha, double *X, const int incX)
{
	AMPBLAS_CHECKED_CALL( ampcblas::scal(N, alpha, X, incX) );
}

void ampblas_cscal(const int N, const ampblas_fcomplex *alpha, ampblas_fcomplex *X, const int incX)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

	const fcomplex calpha = *ampblas_cast(alpha);
	AMPBLAS_CHECKED_CALL( ampcblas::scal(N, calpha, ampblas_cast(X), incX) );
}

void ampblas_zscal(const int N, const ampblas_dcomplex *alpha, ampblas_dcomplex *X, const int incX)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
	AMPBLAS_CHECKED_CALL( ampcblas::scal(N, zalpha, ampblas_cast(X), incX) );
}

void ampblas_csscal(const int N, const float alpha, ampblas_fcomplex *X, const int incX)
{
    using ampcblas::ampblas_cast;

    AMPBLAS_CHECKED_CALL( ampcblas::scal(N, alpha, ampblas_cast(X), incX) );
}

void ampblas_zdscal(const int N, const double alpha, ampblas_dcomplex *X, const int incX)
{
    using ampcblas::ampblas_cast;

	AMPBLAS_CHECKED_CALL( ampcblas::scal(N, alpha, ampblas_cast(X), incX) );
}

} // extern "C"