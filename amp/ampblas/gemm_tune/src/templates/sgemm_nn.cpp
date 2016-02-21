#include "tune.h"
#include "template.h"

TUNE_NAMESPACE_BEGIN

template <>
void do_test<float, transpose::no_trans, transpose::no_trans>(concurrency::accelerator_view& av, float alpha, const concurrency::array_view<const float,2>& a, const concurrency::array_view<const float,2>& b, float beta, const concurrency::array_view<float,2>&c, const std::vector<float>& c_ref, int offset)
{
    #include "../../data/sgemm_nn.data"
}

TUNE_NAMESPACE_END
