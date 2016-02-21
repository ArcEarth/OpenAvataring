#ifndef AMPBLAS_TUNE_DGEMM_H
#define AMPBLAS_TUNE_DGEMM_H

#include "tune.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// ----------------------------------------------------------------------------
// dgemm_nn (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, double, transpose::no_trans, transpose::no_trans>
{
    // dgemm_nn = {32,64,16,32,8,32,8,16,16,1}

    // work block
    static const int m_block = 32; 
    static const int n_block = 64;
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
// dgemm_nt (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, double, transpose::no_trans, transpose::trans>
{
    // dgemm_nt = {32,64,16,32,8,16,16,32,8,1}

    // work block
    static const int m_block = 32; 
    static const int n_block = 64;
    static const int k_block = 16;

    // tile sizes
    static const int m_c_tile = 32;
    static const int n_c_tile = 8;
     
    static const int m_a_tile = 16;
    static const int n_a_tile = 16;
     
    static const int m_b_tile = 32;
    static const int n_b_tile = 8;

    // shared memory padding
    static const int use_padding = 1;
};

// ----------------------------------------------------------------------------
// dgemm_tn (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, double, transpose::trans, transpose::no_trans>
{
    // dgemm_tn = {32,32,8,32,8,8,32,8,32,1}

    // work block
    static const int m_block = 32; 
    static const int n_block = 32;
    static const int k_block = 8;

    // tile sizes
    static const int m_c_tile = 32;
    static const int n_c_tile = 8;
     
    static const int m_a_tile = 8;
    static const int n_a_tile = 32;
     
    static const int m_b_tile = 8;
    static const int n_b_tile = 32;

    // shared memory padding
    static const int use_padding = 1;
};

// ----------------------------------------------------------------------------
// dgemm_tt (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, double, transpose::trans, transpose::trans>
{
    // dgemm_tt = {64,32,16,8,32,8,32,32,8,1}

    // work block
    static const int m_block = 64; 
    static const int n_block = 32;
    static const int k_block = 16;

    // tile sizes
    static const int m_c_tile = 8;
    static const int n_c_tile = 32;
     
    static const int m_a_tile = 8;
    static const int n_a_tile = 32;
     
    static const int m_b_tile = 32;
    static const int n_b_tile = 8;

    // shared memory padding
    static const int use_padding = 1;
};

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_TUNE_DGEMM_H
