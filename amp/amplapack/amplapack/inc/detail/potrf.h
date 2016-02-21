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
* potrf.h
*
*---------------------------------------------------------------------------*/

#ifndef AMPLAPACK_POTRF_H
#define AMPLAPACK_POTRF_H

#include "amplapack_config.h"

// external lapack functions

namespace amplapack {
namespace _detail {

//
// External LAPACK Wrappers
// 

namespace lapack {

template <typename value_type>
void potrf(char uplo, int n, value_type* a, int lda, int& info);

template <>
void potrf(char uplo, int n, float* a, int lda, int& info)
{ 
    LAPACK_SPOTRF(&uplo, &n, a, &lda, &info); 
}

template <>
void potrf(char uplo, int n, double* a, int lda, int& info)
{ 
    LAPACK_DPOTRF(&uplo, &n, a, &lda, &info); 
}

template <>
void potrf(char uplo, int n, ampblas::complex<float>* a, int lda, int& info)
{ 
    LAPACK_CPOTRF(&uplo, &n, a, &lda, &info); 
}

template <>
void potrf(char uplo, int n, ampblas::complex<double>* a, int lda, int& info)
{ 
    LAPACK_ZPOTRF(&uplo, &n, a, &lda, &info); 
}

} // namespace lapack

//
// Host Wrapper
//

namespace host {

template <enum class ordering storage_type, typename value_type>
void potrf(const concurrency::accelerator_view& /*av*/, enum class uplo uplo, concurrency::array_view<value_type,2>& a)
{
    static_assert(storage_type == ordering::column_major, "hybrid functionality requires column major ordering");

    const int n = require_square(a);
   
    // TODO: take from a pool
    const int lda = n;
    std::vector<value_type> hostVector(lda*n);

    // copy from acclerator to host
    concurrency::copy(a, hostVector.begin());

    // run host function
    int info = 0;
    lapack::potrf(to_char(uplo), n, hostVector.data(), lda, info);

    // check for errors
    info_check(info);

    // copy from host to accelerator
    // requires -D_SCL_SECURE_NO_WARNINGS
    concurrency::copy(hostVector.begin(), hostVector.end(), a);
}

} // namespace host

//
// Blocked Factorization
//

template <int block_size, int look_ahead_depth, enum class ordering storage_type, typename value_type>
void potrf(const concurrency::accelerator_view& av, enum class uplo uplo, const concurrency::array_view<value_type,2>& a)
{
    typedef typename ampblas::real_type<value_type>::type real_type;

    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;

    // matrix size
    const int n = require_square(a);

    // stored int lower triangular matrix
    if (uplo == uplo::upper)
    {
        // block stepping
        for (int j = 0; j < n; j += block_size)
        {
            // current block size
            int jb = std::min(block_size, n-j);

            // update diagonal block
            {
                int n_ = jb;
                int k_ = j;

                if (k_ > 0 && n_ > 0)
                {
                    array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(0,j), extent<2>(k_,n_));
                    array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(n_,n_));
                    ampblas::link::herk(av, ampblas::uplo::upper, ampblas::transpose::conj_trans, real_type(-1), a_sub, real_type(1), c_sub);
                }
            }

            // factorize current block
            try
            {
                int n_ = jb;
                array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(n_,n_));
                
                host::potrf<storage_type>(av, uplo, a_sub);
            }
            catch(const data_error_exception& e)
            {
                // offset local block error
                data_error(e.get() + j);
            }

            // this currently has no look ahead optimizations
            if (j+jb < n)
            {
                // compute the current block column
                {
                    int m_ = jb;
                    int n_ = n-j-jb;
                    int k_ = j;
                    if (m_ > 0 && n_ > 0 && k_ > 0)
                    {
                        array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(0,j), extent<2>(k_,m_));
                        array_view<const value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(0,j+jb), extent<2>(k_,n_));
                        array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(a, index<2>(j,j+jb), extent<2>(m_,n_));
                        ampblas::link::gemm(av, ampblas::transpose::conj_trans, ampblas::transpose::no_trans, value_type(-1), a_sub, b_sub, value_type(1), c_sub);
                    }
                }

                // solve for this row
                {
                    int m_ = jb;
                    int n_ = n-j-jb;
                    if (m_ > 0 && n_ > 0)
                    {
                        array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(m_,m_));
                        array_view<value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(j,j+jb), extent<2>(m_,n_));
                        ampblas::link::trsm(av, ampblas::side::left, ampblas::uplo::upper, ampblas::transpose::conj_trans, ampblas::diag::non_unit, value_type(1), a_sub, b_sub);
                    }
                }
            }
        }
    }
    else if (uplo == uplo::lower)
    {
        // block stepping
        for (int j = 0; j < n; j += block_size)
        {
            // current block size
            int jb = std::min(block_size, n-j);

            // update diagonal block
            {
                int n_ = jb;
                int k_ = j;

                if (k_ > 0 && n_ > 0)
                {
                    array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,0), extent<2>(n_,k_));
                    array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(n_,n_));
                    ampblas::link::herk(av, ampblas::uplo::lower, ampblas::transpose::no_trans, real_type(-1), a_sub, real_type(1), c_sub);
                }
            }

            // factorize current block
            try
            {
                int n_ = jb;
                array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(n_,n_));

                host::potrf<storage_type>(av, uplo, a_sub);
            }
            catch(const data_error_exception& e)
            {
                // offset local block error
                data_error(e.get() + j);
            }

            // this currently has no look ahead optimizations
            if (j+jb < n)
            {
                // compute the current block column
                {
                    int m_ = n-j-jb;
                    int n_ = jb;
                    int k_ = j;
                    if (m_ > 0 && n_ > 0 && k_ > 0)
                    {
                        array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j+jb,0), extent<2>(m_,k_));
                        array_view<const value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(j,0), extent<2>(n_,k_));
                        array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(a, index<2>(j+jb,j), extent<2>(m_,n_));
                        ampblas::link::gemm(av, ampblas::transpose::no_trans, ampblas::transpose::conj_trans, value_type(-1), a_sub, b_sub, value_type(1), c_sub);
                    }
                }

                // solve for this row
                {
                    int m_ = n-j-jb;
                    int n_ = jb;
                    if (m_ > 0 && n_ > 0)
                    {
                        array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(n_,n_));
                        array_view<value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(j+jb,j), extent<2>(m_,n_));
                        ampblas::link::trsm(av, ampblas::side::right, ampblas::uplo::lower, ampblas::transpose::conj_trans, ampblas::diag::non_unit, value_type(1), a_sub, b_sub);
                    }
                }
            }
        }
    }
}

} // namespace _detail

//
// Array View Interface
//

template <enum class ordering storage_type, typename value_type>
void potrf(const concurrency::accelerator_view& av, enum class uplo uplo, const concurrency::array_view<value_type,2>& a)
{
    // TODO: a tuning framework
    const int block_size = 256;
    const int look_ahead_depth = 1;

    _detail::potrf<block_size, look_ahead_depth, storage_type>(av, uplo, a);
}

//
// Host Interface Function
//

template <typename value_type>
void potrf(concurrency::accelerator_view& av, char uplo, int n, value_type* a, int lda)
{
    // quick return
    if (n == 0)
        return;
    
    // error checking
    uplo = static_cast<char>(toupper(uplo));

    if (uplo != 'L' && uplo != 'U')
        argument_error(2);
    if (n < 0)
        argument_error(3);
    if (a == nullptr)
        argument_error(4);
    if (lda < n)
        argument_error(5);

    // host views
    concurrency::array_view<value_type,2> host_view_a(n, lda, a);
    concurrency::array_view<value_type,2> host_view_a_sub = host_view_a.section(concurrency::index<2>(0,0), concurrency::extent<2>(n,n));

    // accelerator array (allocation and copy)
    concurrency::array<value_type,2> accl_a(host_view_a_sub);

    // accelerator view
    concurrency::array_view<value_type,2> accl_view_a(accl_a);

    // forwarding function
    potrf<ordering::column_major>(av, to_option(uplo), accl_view_a);

    // copy back to host
    concurrency::copy(accl_view_a, host_view_a_sub);
}

} // namespace amplapack

#endif // AMPLAPACK_POTRF_H
