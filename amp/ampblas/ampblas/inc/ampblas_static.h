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
 * ampblas_static.h
 *
 * Definitions to be included in the static library. Using static definition
 * opposed to pure header definitions will greatly decrease compile time and 
 * and file size if many AMP BLAS calls are required.
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_STATIC_H
#define AMPBLAS_STATIC_H

#include <amp.h>

#include "ampblas_defs.h"
#include "ampblas_complex.h"

namespace ampblas {
namespace link {

//-----------------------------------------------------------------------------
// BLAS 3
//----------------------------------------------------------------------------- 
    
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, float alpha, const concurrency::array_view<const float,2>& a, const concurrency::array_view<const float,2>& b, float beta, const concurrency::array_view<float,2>& c);
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, double alpha, const concurrency::array_view<const double,2>& a, const concurrency::array_view<const double,2>& b, double beta, const concurrency::array_view<double,2>& c);
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, complex<float> alpha, const concurrency::array_view<const complex<float>,2>& a, const concurrency::array_view<const complex<float>,2>& b, complex<float> beta, const concurrency::array_view<complex<float>,2>& c);
void gemm(const concurrency::accelerator_view& av, transpose transa, transpose transb, complex<double> alpha, const concurrency::array_view<const complex<double>,2>& a, const concurrency::array_view<const complex<double>,2>& b, complex<double> beta, const concurrency::array_view<complex<double>,2>& c);

void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, float alpha, const concurrency::array_view<const float,2>& a, const concurrency::array_view<float,2>& b);
void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, double alpha, const concurrency::array_view<const double,2>& a, const concurrency::array_view<double,2>& b);
void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, complex<float> alpha, const concurrency::array_view<const complex<float>,2>& a, const concurrency::array_view<complex<float>,2>& b);
void trsm(const concurrency::accelerator_view& av, side side, uplo uplo, transpose transa, diag diag, complex<double> alpha, const concurrency::array_view<const complex<double>,2>& a, const concurrency::array_view<complex<double>,2>& b);

void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, float  alpha, const concurrency::array_view<const float,2>&           a_mat, float  beta, const concurrency::array_view<float,2>&           c_mat);
void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, double alpha, const concurrency::array_view<const double ,2>&         a_mat, double beta, const concurrency::array_view<double ,2>&         c_mat);
void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, float  alpha, const concurrency::array_view<const complex<float>,2>&  a_mat, float  beta, const concurrency::array_view<complex<float>,2>&  c_mat);
void herk(const concurrency::accelerator_view& av, uplo uplo, transpose trans, double alpha, const concurrency::array_view<const complex<double>,2>& a_mat, double beta, const concurrency::array_view<complex<double>,2>& c_mat);

} // namespace link
} // namespace ampblas

// Visual Studio library link pragma
#ifndef _DEBUG
    #pragma comment(lib,"ampblas_static.lib")
#else
    #pragma comment(lib,"ampblas_staticd.lib")
#endif

#endif // AMPBLAS_STATIC_H