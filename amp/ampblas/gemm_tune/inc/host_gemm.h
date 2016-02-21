#include "tune.h"

TUNE_NAMESPACE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

// external BLAS definitions
void SGEMM(const char*, const char*, const int*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*);
void DGEMM(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
void CGEMM(const char*, const char*, const int*, const int*, const int*, const void*, const void*, const int*, const void*, const int*, const void*, void*, const int*);
void ZGEMM(const char*, const char*, const int*, const int*, const int*, const void*, const void*, const int*, const void*, const int*, const void*, void*, const int*);

#ifdef __cplusplus
} // extern "C" {
#endif

namespace host {

    namespace _detail {
        inline void gemm(char transa, char transb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) { SGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc); }
        inline void gemm(char transa, char transb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) { DGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc); }
        inline void gemm(char transa, char transb, int m, int n, int k, fcomplex alpha, const fcomplex* a, int lda, const fcomplex* b, int ldb, fcomplex beta, fcomplex* c, int ldc) { CGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc); }
        inline void gemm(char transa, char transb, int m, int n, int k, dcomplex alpha, const dcomplex* a, int lda, const dcomplex* b, int ldb, dcomplex beta, dcomplex* c, int ldc) { ZGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc); }
    }

    // row major wrapper
    template <typename value_type>
    void matrix_matrix_multiply(char transa, char transb, int m, int n, int k, value_type alpha, const value_type* a, int lda, const value_type* b, int ldb, value_type beta, value_type* c, int ldc)
    {
        // refactor for column major functions
        _detail::gemm(transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

} // namespace host


TUNE_NAMESPACE_END
