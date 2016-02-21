#ifndef AMPBLAS_TUNE_SGEMM_H
#define AMPBLAS_TUNE_SGEMM_H

#include "tune.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// ----------------------------------------------------------------------------
// sgemm_nn (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, float, transpose::no_trans, transpose::no_trans>
{
    // work block
    static const int m_block = 64; 
    static const int n_block = 48;
    static const int k_block = 32;

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

// ----------------------------------------------------------------------------
// sgemm_nt (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, float, transpose::no_trans, transpose::trans>
{
    // work block
    static const int m_block = 64; 
    static const int n_block = 48;
    static const int k_block = 32;

    // tile sizes
    static const int m_c_tile = 16;
    static const int n_c_tile = 16;
     
    static const int m_a_tile = 32;
    static const int n_a_tile = 8;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

// ----------------------------------------------------------------------------
// sgemm_tn (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, float, transpose::trans, transpose::no_trans>
{
    // work block
    static const int m_block = 32; 
    static const int n_block = 96;
    static const int k_block = 32;

    // tile sizes
    static const int m_c_tile = 16;
    static const int n_c_tile = 16;
     
    static const int m_a_tile = 8;
    static const int n_a_tile = 32;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

// ----------------------------------------------------------------------------
// sgemm_tt (default architecture)
// ----------------------------------------------------------------------------

template <enum class architecture arch>
struct gemm_tuning_parameters<arch, float, transpose::trans, transpose::trans>
{
    // work block
    static const int m_block = 64; 
    static const int n_block = 40;
    static const int k_block = 32;

    // tile sizes
    static const int m_c_tile = 32;
    static const int n_c_tile = 8;
     
    static const int m_a_tile = 4;
    static const int n_a_tile = 64;
     
    static const int m_b_tile = 8;
    static const int n_b_tile = 32;

    // shared memory padding
    static const int use_padding = 1;
};

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_TUNE_SGEMM_H
