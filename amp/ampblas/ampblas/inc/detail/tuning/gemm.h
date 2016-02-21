#ifndef AMPBLAS_TUNE_GEMM_H
#define AMPBLAS_TUNE_GEMM_H

#include "tune.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// ----------------------------------------------------------------------------
// catch all default - will work (though suboptimally) for any configuration
// ----------------------------------------------------------------------------

template <enum class architecture arch, typename value_type, enum class transpose transa, enum class transpose tranbs>
struct gemm_tuning_parameters
{
    // work block
    static const int m_block = 16; 
    static const int n_block = 16;
    static const int k_block = 16;

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

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#include "sgemm.h"
#include "dgemm.h"
#include "cgemm.h"
#include "zgemm.h"

#endif // AMPBLAS_TUNE_GEMM_H
