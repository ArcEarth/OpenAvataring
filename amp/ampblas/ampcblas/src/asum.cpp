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
 * asum.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/asum.h"

namespace ampcblas {

template <typename value_type>
typename ampblas::real_type<value_type>::type asum(const int n, const value_type* x, const int incx)
{
    typedef typename ampblas::real_type<value_type>::type real_type;

    // quick return
	if (n == 0 || incx <= 0)
		return real_type();

	// argument check
	if (x == nullptr)
		argument_error("asum", 2);

    auto x_vec = make_vector_view(n, x, incx);

    return ampblas::asum(get_current_accelerator_view(), x_vec);
}

} // ampcblas

extern "C" {

float ampblas_sasum(const int N, const float *X, const int incX)
{
    float ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::asum(N, X, incX) );
    return ret;
}

double ampblas_dasum(const int N, const double *X, const int incX)
{
    double ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::asum(N, X, incX) );
    return ret;
}

float ampblas_scasum(const int N, const ampblas_fcomplex *X, const int incX)
{
    float ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::asum(N, ampcblas::ampblas_cast(X), incX) );
    return ret;
}

double ampblas_dzasum(const int N, const ampblas_dcomplex *X, const int incX)
{
    double ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::asum(N, ampcblas::ampblas_cast(X), incX) );
    return ret;
}

} // extern "C" 
