// overloaded interfaces (basic tests)
#include "ampxblas.h"

// cblas reference impletnation
#include "cblas_wrapper.h"

// testing headers
#include "ampblas_test_bench.h"
#include "ampblas_test_util.h"

#include <vector>
#include <sstream>

// unique paramaters for ger
template <typename value_type>
struct symm_parameters
{
    symm_parameters(enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, int m, int n, value_type alpha, value_type beta, int lda_offset, int ldb_offset, int ldc_offset)
      : side(side), uplo(uplo), m(m), n(n), alpha(alpha), beta(beta), lda_offset(lda_offset), ldb_offset(ldb_offset), ldc_offset(ldc_offset)
    {}

    enum AMPBLAS_SIDE side;
    enum AMPBLAS_UPLO uplo;
    int m;
    int n;
    value_type alpha;
    value_type beta;
    int lda_offset;
    int ldb_offset;
    int ldc_offset;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(side)
            << AMPBLAS_NAMED_TYPE(uplo)
            << AMPBLAS_NAMED_TYPE(m)
            << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(alpha)
            << AMPBLAS_NAMED_TYPE(beta)
            << AMPBLAS_NAMED_TYPE(lda_offset)
            << AMPBLAS_NAMED_TYPE(ldb_offset)
            << AMPBLAS_NAMED_TYPE(ldc_offset);

        return out.str();
    }

};

template <typename value_type>
class symm_test : public test_case<value_type,symm_parameters>
{
public:

    std::string name() const
    {
        return "SYMM";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // 
        int k = (p.side == AmpblasLeft ? p.m : p.n);

        // derived parameters
        int lda = k + p.lda_offset;
        int ldb = p.m + p.ldb_offset;
        int ldc = p.m + p.ldc_offset;

        // reference data
        test_matrix<value_type> A(k, k, lda);
        test_matrix<value_type> B(p.m, p.n, ldb);
        test_matrix<value_type> C(p.m, p.n, ldc);

        // generate data
        randomize(A);
        randomize(B);
        randomize(C);

        // ampblas bound data
        ampblas_test_matrix<value_type> A_amp(A);
        ampblas_test_matrix<value_type> B_amp(B);
        ampblas_test_matrix<value_type> C_amp(C);

		// cblas types
        cblas::side side = (p.side == AmpblasLeft ? cblas::side::left : cblas::side::right);
		cblas::uplo uplo = (p.uplo == AmpblasUpper ? cblas::uplo::upper : cblas::uplo::lower);

        // test references
        start_reference_test();
        cblas::xSYMM( side, uplo, p.m, p.n, p.alpha, A.data(), A.ld(), B.data(), B.ld(), p.beta, C.data(), C.ld() );
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xsymm( AmpblasColMajor, p.side, p.uplo, p.m, p.n, p.alpha, A_amp.data(), A_amp.ld(), B_amp.data(), B_amp.ld(), p.beta, C_amp.data(), C_amp.ld() );
        stop_ampblas_test();

        // synchronize outputs
        C_amp.synchronize();

        // calculate error
        check_error(C, C_amp);
    }

    symm_test()
    {
        // bulk test
        std::vector<enum AMPBLAS_SIDE> side;
        side.push_back(AmpblasLeft);
        side.push_back(AmpblasRight);

        std::vector<enum AMPBLAS_UPLO> uplo;
        uplo.push_back(AmpblasUpper);
        uplo.push_back(AmpblasLower);

        std::vector<int> m;
        m.push_back(16);
        m.push_back(64);

        std::vector<int> n;
        n.push_back(16);
        n.push_back(64);

        std::vector<value_type> alpha;
        alpha.push_back( value_type(1) );
        alpha.push_back( value_type(-1) );
        alpha.push_back( value_type(0) );

        std::vector<value_type> beta;
        beta.push_back( value_type(2) );
        beta.push_back( value_type(-1) );
        beta.push_back( value_type(0) );

        std::vector<int> lda_offset;
        lda_offset.push_back(0);
        lda_offset.push_back(4);

        std::vector<int> ldb_offset;
        ldb_offset.push_back(0);
        ldb_offset.push_back(4);

        std::vector<int> ldc_offset;
        ldc_offset.push_back(0);
        ldc_offset.push_back(4);

        paramter_exploder( side, uplo, m, n, alpha, beta, lda_offset, ldb_offset, ldc_offset );
    }

    value_type fudge_factor() const { return value_type(20); }
};

REGISTER_TEST(symm_test, float);
REGISTER_TEST(symm_test, double);

