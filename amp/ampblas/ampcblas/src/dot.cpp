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
 * dot.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/dot.h"

namespace ampcblas {

// Generic NRM2 algorithm for AMPBLAS arrays of type T
template <typename value_type, typename accumulation_type, typename trans_op>
accumulation_type dot(int n, const value_type *x, int incx, const value_type *y, int incy)
{
	// quick return
    if (x <= 0) 
        return accumulation_type();
 
    // argument check
    if (x == nullptr)
		argument_error("dot", 2);
    if (y == nullptr)
        argument_error("dot", 4);

    auto x_vec = make_vector_view(n, x, incx);
    auto y_vec = make_vector_view(n, y, incy);

    return ampblas::dot<accumulation_type,trans_op>(get_current_accelerator_view(), x_vec, y_vec);
}

} // namespace ampcblas

extern "C" {

// float ampblas_sdsdot(const int N, const float alpha, const float *X, const int incX, const float *Y, const int incY) 
// {
// }

double ampblas_dsdot(const int N, const float *X, const int incX, const float *Y, const int incY)
{
    double ret = 0;
	AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<float, double, ampblas::_detail::noop>(N,X,incX,Y,incY) );
    return ret;
}

float ampblas_sdot(const int N, const float  *X, const int incX, const float  *Y, const int incY)
{
    float ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<float, float, ampblas::_detail::noop>(N,X,incX,Y,incY) );
    return ret;
}

double ampblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
{   
    double ret = 0;
    AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<double, double, ampblas::_detail::noop>(N,X,incX,Y,incY) );
    return ret;
}

void ampblas_cdotu_sub(const int N, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY, ampblas_fcomplex *dotu)
{
    ampcblas::fcomplex& ret = *ampcblas::ampblas_cast(dotu);
    AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<ampcblas::fcomplex, ampcblas::fcomplex, ampblas::_detail::noop>(N, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

void ampblas_cdotc_sub(const int N, const ampblas_fcomplex *X, const int incX, const ampblas_fcomplex *Y, const int incY, ampblas_fcomplex *dotc)
{
    ampcblas::fcomplex& ret = *ampcblas::ampblas_cast(dotc);
    AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<ampcblas::fcomplex, ampcblas::fcomplex, ampblas::_detail::conjugate>(N, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

void ampblas_zdotu_sub(const int N, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY, ampblas_dcomplex *dotu)
{
    ampcblas::dcomplex& ret = *ampcblas::ampblas_cast(dotu);
    AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<ampcblas::dcomplex, ampcblas::dcomplex, ampblas::_detail::noop>(N, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

void ampblas_zdotc_sub(const int N, const ampblas_dcomplex *X, const int incX, const ampblas_dcomplex *Y, const int incY, ampblas_dcomplex *dotc)
{
    ampcblas::dcomplex& ret = *ampcblas::ampblas_cast(dotc);
    AMPBLAS_CHECKED_CALL( ret = ampcblas::dot<ampcblas::dcomplex, ampcblas::dcomplex, ampblas::_detail::conjugate>(N, ampcblas::ampblas_cast(X), incX, ampcblas::ampblas_cast(Y), incY) );
}

} // extern "C"
