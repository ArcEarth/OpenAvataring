#pragma once
#ifndef AMPLAPACK_TEST_H
#define AMPLAPACK_TEST_H

#include <limits>
#include <memory>

#include "ampblas_complex.h"
#include "ampclapack.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

// using ampblas's types for testing
typedef ampblas::complex<float> fcomplex;
typedef ampblas::complex<double> dcomplex;

// cast from testing types to interface types (real pass through)
template <typename value_type> inline value_type* cast(value_type* ptr) { return ptr; }
template <typename value_type> inline const value_type* cast(const value_type* ptr) { return ptr; }

// cast from testing types to interface types (complex cast)
inline amplapack_fcomplex* cast(fcomplex* ptr) {  return reinterpret_cast<amplapack_fcomplex*>(ptr); }
inline const amplapack_fcomplex* cast(const fcomplex* ptr) { return reinterpret_cast<const amplapack_fcomplex*>(ptr); }
inline amplapack_dcomplex* cast(dcomplex* ptr) { return reinterpret_cast<amplapack_dcomplex*>(ptr); }
inline const amplapack_dcomplex* cast(const dcomplex* ptr) { return reinterpret_cast<const amplapack_dcomplex*>(ptr); }

// test listing
void potrf_test();
void getrf_test();
void geqrf_test();

// LAPACK data type prefix (SDCZ)
template <typename value_type>
char type_prefix();

template <typename value_type>
struct is_complex { static const bool value = false; };

template <typename value_type>
struct is_complex<ampblas::complex<value_type>> { static const bool value = true; };

// matrix-matrix multiply is used in a number of reconstruction algorithms
template <typename value_type>
void gemm(char transa, char transb, int m, int n, int k, value_type alpha, const value_type* a, int lda, const value_type* b, int ldb, value_type beta, value_type* c, int ldc);

// row swap is used in the GETRF reconstruction
template <typename value_type>
void laswp(int n, value_type* a, int lda, int k1, int k2, int* ipiv, int incx);

// q generation from geqrf results
template <typename value_type>
void orgqr(int m, int n, int k, value_type* a, int lda, value_type* tau);

// returns the one-norm of an m by n matrix
template <typename value_type>
typename ampblas::real_type<value_type>::type one_norm(int m, int n, value_type* a, int lda)
{
    typedef ampblas::real_type<value_type>::type real_type;

    real_type norm = real_type();

    for (int j = 0; j < n; j++)
    {
        real_type sum = real_type();
        for (int i = 0; i < m; i++)
            sum += abs( a[j*lda+i] );

        norm = std::max(norm, sum);
    }

    return norm / n;
}

// 
template <typename value_type>
value_type random_value(value_type min, value_type max)
{
    value_type val = value_type(rand()) / value_type(RAND_MAX);
    val *= (max-min);
    val += min;
    return val;
}

template <typename value_type>
typename ampblas::complex<value_type> random_value( ampblas::complex<value_type> min,  ampblas::complex<value_type> max)
{
    return ampblas::complex<value_type>(random_value(min.real(),max.real()), random_value(min.real(),max.real()));
}

// high resolution timer for use on Windows platforms
class high_resolution_timer
{
public:
    high_resolution_timer();
    ~high_resolution_timer();

    void restart();
    double elapsed();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

#endif // AMPLAPACK_TEST_H
