#include "tune.h"
#include "template.h"

TUNE_NAMESPACE_BEGIN

template <>
void do_test<double, transpose::trans, transpose::trans>(concurrency::accelerator_view& av, double alpha, const concurrency::array_view<const double,2>& a, const concurrency::array_view<const double,2>& b, double beta, const concurrency::array_view<double,2>&c, const std::vector<double>& c_ref, int offset)
{
    #include "../../data/dgemm_tt.data"
}

TUNE_NAMESPACE_END
