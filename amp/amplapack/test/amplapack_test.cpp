#include <assert.h>

#include "amplapack_test.h"
#include "lapack_host.h"

#include "ampclapack.h"

// LAPACK data type prefix (SDCZ)
template<> char type_prefix<float>() { return 'S'; }
template<> char type_prefix<double>() { return 'S'; }
template<> char type_prefix<fcomplex>() { return 'C'; }
template<> char type_prefix<dcomplex>() { return 'Z'; }

// GEMM
template <>
void gemm(char transa, char transb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc)
{
    LAPACK_SGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

template <>
void gemm(char transa, char transb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc)
{
    LAPACK_DGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

template <>
void gemm(char transa, char transb, int m, int n, int k, fcomplex alpha, const fcomplex* a, int lda, const fcomplex* b, int ldb, fcomplex beta, fcomplex* c, int ldc)
{
    LAPACK_CGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

template <>
void gemm(char transa, char transb, int m, int n, int k, dcomplex alpha, const dcomplex* a, int lda, const dcomplex* b, int ldb, dcomplex beta, dcomplex* c, int ldc)
{
    LAPACK_ZGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

// LASWP
template <>
void laswp(int n, float* a, int lda, int k1, int k2, int* ipiv, int incx)
{
    LAPACK_SLASWP(&n, a, &lda, &k1, &k2, ipiv, &incx);
}

template <>
void laswp(int n, double* a, int lda, int k1, int k2, int* ipiv, int incx)
{
    LAPACK_DLASWP(&n, a, &lda, &k1, &k2, ipiv, &incx);
}

template <>
void laswp(int n, fcomplex* a, int lda, int k1, int k2, int* ipiv, int incx)
{
    LAPACK_CLASWP(&n, a, &lda, &k1, &k2, ipiv, &incx);
}

template <>
void laswp(int n, dcomplex* a, int lda, int k1, int k2, int* ipiv, int incx)
{
    LAPACK_ZLASWP(&n, a, &lda, &k1, &k2, ipiv, &incx);
}

// ORGQR
template <>
void orgqr(int m, int n, int k, float* a, int lda, float* tau)
{
    // work query
    int info;
    int lwork = -1;
    float work_size;
    LAPACK_SORGQR(&m, &m, &k, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size);
    std::vector<float> work(lwork);

    LAPACK_SORGQR(&m, &m, &k, a, &lda, tau, work.data(), &lwork, &info);
    assert(info == 0);
}

template <>
void orgqr(int m, int n, int k, double* a, int lda, double* tau)
{
    // work query
    int info;
    int lwork = -1;
    double work_size;
    LAPACK_DORGQR(&m, &m, &k, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size);
    std::vector<double> work(lwork);

    LAPACK_DORGQR(&m, &m, &k, a, &lda, tau, work.data(), &lwork, &info);
    assert(info == 0);
}

template <>
void orgqr(int m, int n, int k, fcomplex* a, int lda, fcomplex* tau)
{
    // work query
    int info;
    int lwork = -1;
    fcomplex work_size;
    LAPACK_CUNGQR(&m, &m, &k, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size.real());
    std::vector<fcomplex> work(lwork);

    LAPACK_CUNGQR(&m, &m, &k, a, &lda, tau, work.data(), &lwork, &info);
    assert(info == 0);
}

template <>
void orgqr(int m, int n, int k, dcomplex* a, int lda, dcomplex* tau)
{
    // work query
    int info;
    int lwork = -1;
    dcomplex work_size;
    LAPACK_ZUNGQR(&m, &m, &k, a, &lda, tau, &work_size, &lwork, &info);
    lwork = int(work_size.real());
    std::vector<dcomplex> work(lwork);

    LAPACK_ZUNGQR(&m, &m, &k, a, &lda, tau, work.data(), &lwork, &info);
    assert(info == 0);
}

int main()
{
    potrf_test();
    getrf_test();
    geqrf_test();
}
