#ifndef AMPBLAS_TUNE_ZGEMM_H
#define AMPBLAS_TUNE_ZGEMM_H

#include "tune.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// ----------------------------------------------------------------------------
// cgemm_nn (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::no_trans, transpose::no_trans>
{
    // zgemm_nn = {32,16,16,32,8,32,8,16,16,1}

    // work block
    static const int m_block = 32; 
    static const int n_block = 16;
    static const int k_block = 16;

    // tile sizes
    static const int m_c_tile = 32;
    static const int n_c_tile = 8;
     
    static const int m_a_tile = 32;
    static const int n_a_tile = 8;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

// ----------------------------------------------------------------------------
// cgemm_nt & cgemm_nc (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::no_trans, transpose::trans>
{
    // zgemm_nt = {32,16,16,32,8,16,16,16,16,1}

    // work block
    static const int m_block = 32; 
    static const int n_block = 16;
    static const int k_block = 16;

    // tile sizes
    static const int m_c_tile = 32;
    static const int n_c_tile = 8;
     
    static const int m_a_tile = 16;
    static const int n_a_tile = 16;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::no_trans, transpose::conj_trans> :
       gemm_tuning_parameters<arch, complex<double>, transpose::no_trans, transpose::trans     >
{};

// ----------------------------------------------------------------------------
// cgemm_tn & cgemm_cn (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::trans, transpose::no_trans>
{
    // zgemm_tn = {16,16,48,16,16,16,16,16,16,1}

    // work block
    static const int m_block = 16; 
    static const int n_block = 16;
    static const int k_block = 48;

    // tile sizes
    static const int m_c_tile = 16;
    static const int n_c_tile = 16;
     
    static const int m_a_tile = 16;
    static const int n_a_tile = 16;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::conj_trans, transpose::no_trans> :
       gemm_tuning_parameters<arch, complex<double>, transpose::trans,      transpose::no_trans>
{};

// ----------------------------------------------------------------------------
// cgemm_tt (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::trans, transpose::trans>
{
    // zgemm_tt = {32,16,16,32,8,16,16,16,16,1}

    // work block
    static const int m_block = 32; 
    static const int n_block = 16;
    static const int k_block = 16;

    // tile sizes
    static const int m_c_tile = 32;
    static const int n_c_tile = 8;
     
    static const int m_a_tile = 16;
    static const int n_a_tile = 16;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, complex<double>, transpose::conj_trans, transpose::conj_trans> :
       gemm_tuning_parameters<arch, complex<double>, transpose::trans,      transpose::trans     >
{};

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_TUNE_ZGEMM_H
