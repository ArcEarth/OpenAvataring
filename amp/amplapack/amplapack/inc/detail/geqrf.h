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
 * geqrf.h
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPLAPACK_GEQRF_H
#define AMPLAPACK_GEQRF_H

#include "amplapack_config.h"

// external lapack functions
namespace amplapack {
namespace _detail {

//
// External LAPACK Wrappers
// 

namespace lapack {

template <typename value_type>
void geqrf(int m, int n, value_type* a, int lda, value_type* tau, int& info);

template <>
void geqrf<float>(int m, int n, float* a, int lda, float* tau, int& info)
{ 
    // work query
    int lwork = -1;
    float work_size;
    LAPACK_SGEQRF(&m, &n, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size);
    std::vector<float> work(lwork);

    LAPACK_SGEQRF(&m, &n, a, &lda, tau, work.data(), &lwork, &info); 
}

template <>
void geqrf<double>(int m, int n, double* a, int lda, double* tau, int& info)
{ 
    // work query
    int lwork = -1;
    double work_size;
    LAPACK_DGEQRF(&m, &n, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size);
    std::vector<double> work(lwork);

    LAPACK_DGEQRF(&m, &n, a, &lda, tau, work.data(), &lwork, &info); 
}

// template <>
void geqrf(int m, int n, ampblas::complex<float>* a, int lda, ampblas::complex<float>* tau, int& info)
{
    // work query
    int lwork = -1;
    ampblas::complex<float> work_size;
    LAPACK_CGEQRF(&m, &n, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size.real());
    std::vector<ampblas::complex<float>> work(lwork);

    LAPACK_CGEQRF(&m, &n, a, &lda, tau, work.data(), &lwork, &info);
}

// template <>
void geqrf(int m, int n, ampblas::complex<double>* a, int lda, ampblas::complex<double>* tau, int& info)
{
    // work query
    int lwork = -1;
    ampblas::complex<double> work_size;
    LAPACK_ZGEQRF(&m, &n, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size.real());
    std::vector<ampblas::complex<double>> work(lwork);

    LAPACK_ZGEQRF(&m, &n, a, &lda, tau, work.data(), &lwork, &info);
}

template <typename value_type>
void larft(char direct, char storev, int n, int k, value_type* v, int ldv, value_type* tau, value_type* t, int ldt);

template <>
void larft(char direct, char storev, int n, int k, float* v, int ldv, float* tau, float* t, int ldt)
{
    LAPACK_SLARFT(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
}

template <>
void larft(char direct, char storev, int n, int k, double* v, int ldv, double* tau, double* t, int ldt)
{
    LAPACK_DLARFT(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
}

template <>
void larft(char direct, char storev, int n, int k, ampblas::complex<float>* v, int ldv, ampblas::complex<float>* tau, ampblas::complex<float>* t, int ldt)
{
    LAPACK_CLARFT(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
}

template <>
void larft(char direct, char storev, int n, int k, ampblas::complex<double>* v, int ldv,  ampblas::complex<double>* tau,  ampblas::complex<double>* t, int ldt)
{
    LAPACK_ZLARFT(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
}

} // namespace lapack

//
// Host Wrapper
//

namespace host {

template <enum class ordering storage_type, typename value_type>
void geqrf(const concurrency::accelerator_view& /*av*/, concurrency::array_view<value_type,2>& a, concurrency::array_view<value_type,1>& tau)
{
    static_assert(storage_type == ordering::column_major, "hybrid functionality requires column major ordering");

    const int m = get_rows<storage_type>(a);
    const int n = get_cols<storage_type>(a);
   
    // TODO: take from a pool
    const int lda = get_leading_dimension<storage_type>(a);
    std::vector<value_type> host_a(lda*n);

    // copy from acclerator to host
    concurrency::copy(a, host_a.begin());

    // run host function
    int info = 0;
    lapack::geqrf(m, n, host_a.data(), lda, tau.data(), info);

    // check for errors
    info_check(info);

    // copy from host to accelerator
    // requires -D_SCL_SECURE_NO_WARNINGS
    concurrency::copy(host_a.begin(), host_a.end(), a);
}

template <enum class ordering storage_type, typename value_type>
void larft(const concurrency::accelerator_view& /*av*/, enum class direction /*direct*/, enum class storage storev, concurrency::array_view<value_type,2>& v, concurrency::array_view<value_type,1>& tau, concurrency::array_view<value_type,2>& t)
{
    static_assert(storage_type == ordering::column_major, "hybrid functionality requires column major ordering");

    const int n = (storev == storage::column ? get_rows<storage_type>(v) : get_cols<storage_type>(v));
    const int k = (storev == storage::column ? get_cols<storage_type>(v) : get_rows<storage_type>(v));

    // host v
    const int ldv = n;
    std::vector<value_type> host_v(ldv*k);
    concurrency::copy(v, host_v.begin());

    // host t
    const int ldt = k;
    std::vector<value_type> host_t(ldt*k);
    concurrency::copy(t, host_t.begin());

    // run host function
    lapack::larft('f', 'c', n, k, host_v.data(), ldv, tau.data(), host_t.data(), ldt);

    // copy from host to accelerator
    // requires -D_SCL_SECURE_NO_WARNINGS
    concurrency::copy(host_t.begin(), host_t.end(), t);
}

} // namespace host

//
// Accelerator Helper Functions
//

template <enum class ordering storage_type, typename value_type>
void make_unit_lower(const concurrency::accelerator_view& av, concurrency::array_view<value_type,2>& a)
{
    concurrency::parallel_for_each(av, a.extent, [=] (concurrency::index<2> idx) restrict(amp) 
    {
        // TODO: this is only column major
        unsigned int i = idx[1];
        unsigned int j = idx[0];

        if (i < j)
        {
            a[idx] = value_type();
        }
        else if (i == j)
        {
            a[idx] = value_type(1);
        }
    });
}

//
// Blocked Factorization
//

template <int block_size, int look_ahead_depth, enum class ordering storage_type, enum class block_factor_location location, typename value_type>
void geqrf(const concurrency::accelerator_view& av, concurrency::array_view<value_type,2>& a, concurrency::array_view<value_type,1>& tau)
{
    using concurrency::array;
    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;
    
    // sizes
    const int m = get_rows<storage_type>(a);
    const int n = get_cols<storage_type>(a);
    const int k = std::min(m,n);

    // working array for v1 storage (accelerator)
    // TODO: not needed for host-only interface
    array<value_type,2> array_v1(block_size, block_size);
    array_view<value_type,2> v1(array_v1);

    // working array for triangular factor (accelerator)
    array<value_type,2> array_t(block_size, block_size);
    array_view<value_type,2> t(array_t);

    // working array for w (accelerator)
    // TODO: this is only column major
    array<value_type,2> array_w(n, block_size);
    array_view<value_type,2> w(array_w);

    // working array for w_t (accelerator)
    // TODO: this is only column major
    array<value_type,2> array_wt(block_size, m);
    array_view<value_type,2> wt(array_wt);

    // panel stepping
    for (int i = 0; i < k; i += block_size)
    {
        // current panel size
        const int ib = std::min(k-i, block_size);

        // panel factorization
        {
            int m_ = m-i;
            int n_ = ib;

            array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(i,i), extent<2>(m_,n_)); 
            array_view<value_type,1> tau_sub = tau.section(index<1>(i)); 

            host::geqrf<storage_type>(av, a_sub, tau_sub);
        }

        // apply to rest of matrix (no look ahead yet)
        if (i+ib < n)
        {
            //
            // A2 = Q' * A2 = (I - V * T' * V') * A2
            //

            // form the triangular factor (t) of the block reflector
            {
                int m_ = m-i;
                int n_ = ib;

                array_view<value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(i,i), extent<2>(m_,n_));
                array_view<value_type,1> tau_sub = tau.section(index<1>(i));
                array_view<value_type,2> t_sub = t.section(index<2>(0,0), extent<2>(ib,ib));
                                
                host::larft<storage_type>(av, direction::forward, storage::column, a_sub, tau_sub, t_sub);
            }

            // backup v1 (not needed for host-only interface)
            {
                array_view<value_type,2> a_sub = a.section(index<2>(i,i), extent<2>(ib,ib));
                array_view<value_type,2> v1_sub = v1.section(index<2>(0,0), extent<2>(ib,ib));

                concurrency::copy(a_sub, v1_sub);
            }

            // make v unit-lower
            {
                array_view<value_type,2> a_sub = a.section(index<2>(i,i), extent<2>(ib,ib));
                make_unit_lower<storage_type>(av, a_sub);
            }

            // w = t' * v' <==> w' = v * t 
            {
                int m_ = m-i;
                int n_ = ib;
                int k_ = ib;

                array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(i,i), extent<2>(m_,k_));
                array_view<const value_type,2> b_sub = get_sub_matrix<storage_type>(t, index<2>(0,0), extent<2>(k_,n_));
                array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(wt, index<2>(0,0), extent<2>(m_,n_));

                ampblas::link::gemm(av, ampblas::transpose::no_trans, ampblas::transpose::no_trans, value_type(1), a_sub, b_sub, value_type(), c_sub);
            }

            // w = w' * a2
            {
                int m_ = ib;
                int n_ = n-i-ib;
                int k_ = m-i;

                array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(wt, index<2>(0,0), extent<2>(k_,m_));
                array_view<const value_type,2> b_sub = get_sub_matrix<storage_type>(a, index<2>(i,i+ib), extent<2>(k_,n_));
                array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(w, index<2>(0,0), extent<2>(m_,n_));

                ampblas::link::gemm(av, ampblas::transpose::conj_trans, ampblas::transpose::no_trans, value_type(1), a_sub, b_sub, value_type(), c_sub);
            }

            // a2 -= v * w
            {
                int m_ = m-i;
                int n_ = n-i-ib;
                int k_ = ib;

                array_view<const value_type,2> a_sub = get_sub_matrix<storage_type>(a, index<2>(i,i), extent<2>(m_,k_));
                array_view<const value_type,2> b_sub = get_sub_matrix<storage_type>(w, index<2>(0,0), extent<2>(k_,n_));
                array_view<value_type,2> c_sub = get_sub_matrix<storage_type>(a, index<2>(i,i+ib), extent<2>(m_,n_));

                ampblas::link::gemm(av, ampblas::transpose::no_trans, ampblas::transpose::no_trans, value_type(-1), a_sub, b_sub, value_type(1), c_sub);
            }

            // restore v1 (not needed for host-only interface)
            {
                array_view<value_type,2> a_sub = a.section(index<2>(i,i), extent<2>(ib,ib));
                array_view<value_type,2> v1_sub = v1.section(index<2>(0,0), extent<2>(ib,ib));

                concurrency::copy(v1_sub, a_sub);
            }
        }
    }
}

// this is a work around until VS std::bind can accept more paramaters
template <typename value_type>
struct geqrf_params
{
    int m;
    int n; 
    value_type* a; 
    int lda; 
    value_type* tau;

    geqrf_params(int m, int n, value_type* a, int lda, value_type* tau)
        : m(m), n(n), a(a), lda(lda), tau(tau) 
    {}
};

template <typename value_type>
void geqrf_unpack(concurrency::accelerator_view& av, const geqrf_params<value_type>& p)
{
    amplapack::geqrf(av, p.m, p.n, p.a, p.lda, p.tau); 
}

} // namespace _detail

//
// Array View Interface
//

template <enum class ordering storage_type, typename value_type>
void geqrf(const concurrency::accelerator_view& av, concurrency::array_view<value_type,2>& a, concurrency::array_view<value_type,1>& tau)
{
    // TODO: a tuning framework
    const int block_size = 256;
    const int look_ahead_depth = 1;

    _detail::geqrf<block_size, look_ahead_depth, storage_type, block_factor_location::host>(av, a, tau);
}

//
// Host Interface Function
//

template <typename value_type>
void geqrf(concurrency::accelerator_view& av, int m, int n, value_type* a, int lda, value_type* tau)
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
    if (tau == nullptr)
        argument_error(6);

    // host views
    concurrency::array_view<value_type,2> host_view_a(n, lda, a);
    concurrency::array_view<value_type,2> host_view_a_sub = host_view_a.section(concurrency::index<2>(0,0), concurrency::extent<2>(n,m));
    concurrency::array_view<value_type,1> host_view_tau(std::min(m,n), tau);

    // accelerator array (allocation and copy)
    concurrency::array<value_type,2> accl_a(host_view_a_sub);

    // accelerator view
    concurrency::array_view<value_type,2> accl_view_a(accl_a);

    // foward to array view interface
    geqrf<ordering::column_major>(av, accl_view_a, host_view_tau);

    // copy back to host
    concurrency::copy(accl_view_a, host_view_a_sub);
}

} // namespace amplapack

#endif // AMPLAPACK_GEQRF_H
