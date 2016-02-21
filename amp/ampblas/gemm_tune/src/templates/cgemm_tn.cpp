#include "tune.h"
#include "template.h"

TUNE_NAMESPACE_BEGIN

template <>
void do_test<fcomplex, transpose::trans, transpose::no_trans>(concurrency::accelerator_view& av, fcomplex alpha, const concurrency::array_view<const fcomplex,2>& a, const concurrency::array_view<const fcomplex,2>& b, fcomplex beta, const concurrency::array_view<fcomplex,2>&c, const std::vector<fcomplex>& c_ref, int offset)
{
    #include "../../data/cgemm_tn.data"
}

TUNE_NAMESPACE_END
