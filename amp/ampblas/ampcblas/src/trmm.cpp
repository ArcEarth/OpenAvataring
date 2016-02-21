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
 * trmm.cpp
 *
 *---------------------------------------------------------------------------*/

#include "ampcblas_config.h"

#include "detail/trmm.h"

namespace ampcblas {

template <typename value_type>
void trmm(enum AMPBLAS_ORDER order, enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, const value_type* a, int lda, value_type* b, int ldb)
{
    // recursive order adjustment
    if (order == AmpblasRowMajor) 
    {
        trmm(AmpblasColMajor, side == AmpblasLeft ? AmpblasRight : AmpblasLeft, uplo == AmpblasUpper ? AmpblasLower : AmpblasUpper, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }

    // quick return
    if (m == 0 && n == 0) 
        return;

    // derived parameters
    int k = (side == AmpblasLeft ? m : n);

    // argument check
    if (m < 0)
        argument_error("trmm", 6);
    if (n < 0)
        argument_error("trmm", 7);
    if (a == nullptr)
        argument_error("trmm", 9);
    if (lda < k)
        argument_error("trmm", 10);
    if (b == nullptr)
        argument_error("trmm", 11);
    if (ldb < m)
        argument_error("trmm", 12);

    // create views
    auto a_mat = make_matrix_view(k, k, a, lda);
    auto b_mat = make_matrix_view(m, n, b, ldb);
    auto b_mat_const = make_matrix_view(m, n, const_cast<const value_type*>(b), ldb);

    // fill with zeros if alpha is zero
    if (alpha == value_type())
    {
        ampblas::_detail::fill(get_current_accelerator_view(), b_mat.extent, value_type(), b_mat);
        return;
    }

    // workspace
    concurrency::array<value_type,2> c(n,m); 
    concurrency::array_view<value_type,2> c_mat(c);
    c_mat.discard_data();

    // forward to tuning routine
    ampblas::trmm(get_current_accelerator_view(), cast(side), cast(uplo), cast(transa), cast(diag), m, n, alpha, a_mat, b_mat_const, c_mat);

    // copy workspace to answer
    copy(c_mat, b_mat);
}

} // namespace ampcblas

extern "C" {

void ampblas_strmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb)
{
    AMPBLAS_CHECKED_CALL( ampcblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) );
}
void ampblas_dtrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb)
{
    AMPBLAS_CHECKED_CALL( ampcblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) );
}

void ampblas_ctrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_fcomplex *alpha, const ampblas_fcomplex *A, const int lda, ampblas_fcomplex *B, const int ldb)
{
    using ampcblas::fcomplex;
    using ampcblas::ampblas_cast;

    const fcomplex calpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, calpha, ampblas_cast(A), lda, ampblas_cast(B), ldb) );
}
void ampblas_ztrmm(const enum AMPBLAS_ORDER Order, const enum AMPBLAS_SIDE Side, const enum AMPBLAS_UPLO Uplo, const enum AMPBLAS_TRANSPOSE TransA, const enum AMPBLAS_DIAG Diag, const int M, const int N, const ampblas_dcomplex *alpha, const ampblas_dcomplex *A, const int lda, ampblas_dcomplex *B, const int ldb)
{
    using ampcblas::dcomplex;
    using ampcblas::ampblas_cast;

    const dcomplex zalpha = *ampblas_cast(alpha);
    AMPBLAS_CHECKED_CALL( ampcblas::trmm(Order, Side, Uplo, TransA, Diag, M, N, zalpha, ampblas_cast(A), lda, ampblas_cast(B), ldb) );
}

} // extern "C"
