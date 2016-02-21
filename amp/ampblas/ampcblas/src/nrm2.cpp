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
 * nrm2.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/nrm2.h"

namespace ampcblas {

// Generic NRM2 algorithm for AMPBLAS arrays of type T
template <typename value_type>
value_type nrm2(int n, const value_type *x, int incx)
{
	// quick return
	if (n <= 0) 
        return value_type();

    // check arguments
    if (x == nullptr)
		argument_error("nrm2", 2);
        
    auto x_vec = make_vector_view(n, x, incx);
	return ampblas::nrm2(get_current_accelerator_view(), x_vec);
}

} // namespace ampcblas

extern "C" {

float ampblas_snrm2(const int N, const float* X, int incX )
{
    float ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::nrm2(N, X, incX) );
    return ret;
}

double ampblas_dnrm2(const int N, const double* X, int incX )
{
    double ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::nrm2(N, X, incX) );
    return ret;
}

} // extern "C"
