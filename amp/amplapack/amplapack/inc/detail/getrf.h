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
* getrf.h
*
*---------------------------------------------------------------------------*/

#ifndef AMPLAPACK_GETRF_H
#define AMPLAPACK_GETRF_H

#include "amplapack_config.h"

// external lapack functions

namespace amplapack {
namespace _detail {

//
// External LAPACK Wrappers
// 

namespace lapack {

template <typename value_type>
void getrf(int m, int n, value_type* a, int lda, int* ipiv, int& info);

template <>
void getrf(int m, int n, float* a, int lda, int* ipiv, int& info)
{ 
    LAPACK_SGETRF(&m, &n, a, &lda, ipiv, &info); 
}

template <>
void getrf(int m, int n, double* a, int lda, int* ipiv, int& info)
{ 
    LAPACK_DGETRF(&m, &n, a, &lda, ipiv, &info); 
}

template <>
void getrf(int m, int n, ampblas::complex<float>* a, int lda, int* ipiv, int& info)
{ 
    LAPACK_CGETRF(&m, &n, a, &lda, ipiv, &info); 
}

template <>
void getrf(int m, int n, ampblas::complex<double>* a, int lda, int* ipiv, int& info)
{
    LAPACK_ZGETRF(&m, &n, a, &lda, ipiv, &info); 
}

} // namespace lapack

//
// Host Wrapper
//

namespace host {

template <enum class ordering storage_type, typename value_type>
void getrf(const concurrency::accelerator_view& /*av*/, concurrency::array_view<value_type,2>& a, concurrency::array_view<int,1>& ipiv)
{
    static_assert(storage_type == ordering::column_major, "hybrid functionality requires column major ordering");

    const int m = get_rows<storage_type>(a);
    const int n = get_cols<storage_type>(a);
   
    // TODO: take from a pool
    const int lda = get_leading_dimension<storage_type>(a);
    std::vector<value_type> hostVector(lda*n);

    // copy from acclerator to host
    concurrency::copy(a, hostVector.begin());

    // run host function
    int info = 0;
    lapack::getrf(m, n, hostVector.data(), lda, ipiv.data(), info);

    // check for errors
    info_check(info);

    // copy from host to accelerator
    // requires -D_SCL_SECURE_NO_WARNINGS
    concurrency::copy(hostVector.begin(), hostVector.end(), a);
}

} // namespace host

//
// Row Interchanges
//

template <typename T>
inline void swap(T& a, T& b) restrict(amp)
{
    T temp = a;
    a = b;
    b = temp;
}

template <enum class ordering storage_type, typename value_type>
void laswp(const concurrency::accelerator_view& av, concurrency::array_view<value_type,2>& a, int k1, int k2, concurrency::array_view<int,1>& ipiv)
{
    using concurrency::index;

    const int n = get_cols<storage_type>(a);

    // only forward swaps are implemented (incx >= 1)
    concurrency::parallel_for_each(
        av,
        concurrency::extent<1>(n),
        [=](index<1> idx) restrict(amp) 
        {
            const int j = idx[0];

            for (int i = k1; i < k2; i++)
            {
                // pivot index (Fortran indexing)
                int ip = ipiv[index<1>(i)]-1;
                
                if (i != ip)
                {
                    swap(a(index<2>(j,ip)), a(index<2>(j,i)));
                }
            }
        }
    );   
}

//
// Blocked Factorization
//

template <int block_size, int look_ahead_depth, enum class ordering storage_type, enum class block_factor_location location, typename value_type>
void getrf(const concurrency::accelerator_view& av, concurrency::array_view<value_type,2>& a, concurrency::array_view<int,1>& ipiv)
{
    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;

    // data error
    int info = 0;
    
    // sizes
    const int m = get_rows<storage_type>(a);
    const int n = get_cols<storage_type>(a);
    const int k = std::min(m,n);

    // panel stepping
    for (int j = 0; j < k; j += block_size)
    {
        // current block size
        int jb = std::min(block_size, k-j);

        // factor diagonal and subdiagonal blocks and test for exact singularity
        try 
        {
            int m_ = m-j;
            int n_ = jb;
            int k_ = std::min(m_,n_);
            array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(m_,n_)); 
            array_view<int,1> ipiv_sub = ipiv.section(index<1>(j), extent<1>(k_)); 
            host::getrf<storage_type>(av, a_sub, ipiv_sub);
        }
        catch(const data_error_exception& e)
        {
            // offset data error (do not rethrow) 
            info = j + e.get();
        }
        
        // offset pivot vector
        for (int i=0; i<jb; i++)
        {
            ipiv(index<1>(j+i)) += j;
        }

        // apply interchanges to columns 1:j
        if (j > 0)
        {
            int m_ = m;
            int n_ = j;
            array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(0,0), extent<2>(m_,n_));

            laswp<storage_type>(av, a_sub, j, j+jb, ipiv);
        }

        // apply to rest of matrix
        if (j+jb < n)
        {
            // apply interchange to columns j+jb:n
            {
                int m_ = m;
                int n_ = n-j-jb;
                array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(0,j+jb), extent<2>(m_,n_));

                laswp<storage_type>(av, a_sub, j, j+jb, ipiv);
            }

            // compute block row of U
            {
                int m_ = jb;
                int n_ = n-j-jb;

                array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j,j), extent<2>(m_,m_));
                array_view<value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(j,j+jb), extent<2>(m_,n_));

                ampblas::link::trsm(av, ampblas::side::left, ampblas::uplo::lower, ampblas::transpose::no_trans, ampblas::diag::unit, value_type(1), a_sub, b_sub);
            }

            // no look ahead yet
            if (j+jb < m)
            {
                // update trailing matrix
                int m_ = m-j-jb;
                int n_ = n-j-jb;
                int k_ = jb;

                array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(j+jb,j), extent<2>(m_,k_));
                array_view<const value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(j,j+jb), extent<2>(k_,n_));
                array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(a, index<2>(j+jb,j+jb), extent<2>(m_,n_));

                ampblas::link::gemm(av, ampblas::transpose::no_trans, ampblas::transpose::no_trans, value_type(-1), a_sub, b_sub, value_type(1), c_sub);
            }
        }
    }

    // rethrow data error (if any)
    if (info)
        data_error(info);
}

//
// Forwarding Function
//

// this is a work around until VS std::bind can accept more paramaters
template <typename value_type>
struct getrf_params
{
    int m;
    int n; 
    value_type* a; 
    int lda; 
    int* ipiv;

    getrf_params(int m, int n, value_type* a, int lda, int* ipiv)
        : m(m), n(n), a(a), lda(lda), ipiv(ipiv) 
    {}
};

template <typename value_type>
void getrf_unpack(concurrency::accelerator_view& av, const getrf_params<value_type>& p)
{
    amplapack::getrf(av, p.m, p.n, p.a, p.lda, p.ipiv); 
}

} // namespace _detail

//
// Array View Interface
// 

template <enum class ordering storage_type, typename value_type>
void getrf(const concurrency::accelerator_view& av, concurrency::array_view<value_type,2>& a, concurrency::array_view<int,1>& ipiv)
{
    const int block_size = 256;
    const int look_ahead_depth = 1;

    _detail::getrf<block_size, look_ahead_depth, storage_type, block_factor_location::host>(av, a, ipiv);
}

//
// Host Interface Function
//

template <typename value_type>
void getrf(concurrency::accelerator_view& av, int m, int n, value_type* a, int lda, int* ipiv)
{
    // quick return
    if (n == 0 || m == 0)
        return;

    // error checking
    if (m < 0)
        argument_error(2);
    if (n < 0)
        argument_error(3);
    if (a == nullptr)
        argument_error(4);
    if (lda < m)
        argument_error(5);
    if (ipiv == nullptr)
        argument_error(6);

    // host views
    concurrency::array_view<value_type,2> host_view_a(n, lda, a);
    concurrency::array_view<value_type,2> host_view_a_sub = host_view_a.section(concurrency::index<2>(0,0), concurrency::extent<2>(n,m));
    concurrency::array_view<int,1> host_view_ipiv(std::min(m,n), ipiv);

    // accelerator array (allocation and copy)
    concurrency::array<value_type,2> accl_a(host_view_a_sub);

    // accelerator view
    concurrency::array_view<value_type,2> accl_view_a(accl_a);

    // forwarding to array view interface
    getrf<ordering::column_major>(av, accl_view_a, host_view_ipiv);

    // copy back to host
    concurrency::copy(accl_view_a, host_view_a_sub);
}

} // namespace amplapack

#endif // AMPLAPACK_GETRF_H
