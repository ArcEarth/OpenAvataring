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
struct syr2k_parameters
{
    syr2k_parameters(enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, value_type alpha, value_type beta, int lda_offset, int ldb_offset, int ldc_offset)
      : uplo(uplo), trans(trans), n(n), k(k), alpha(alpha), beta(beta), lda_offset(lda_offset), ldb_offset(ldb_offset), ldc_offset(ldc_offset)
    {}

    enum AMPBLAS_UPLO uplo;
    enum AMPBLAS_TRANSPOSE trans;int n;
    int k;
    value_type alpha;
    value_type beta;
    int lda_offset;
    int ldb_offset;
    int ldc_offset;

    std::string name() const
    {
        std::stringstream out;

        char uplo = (this->uplo==AmpblasUpper)?'U':'L';
        char trans = (this->trans==AmpblasNoTrans)?'N':((this->trans==AmpblasTrans)?'T':'C');
        
        out << AMPBLAS_NAMED_TYPE(uplo)
            << AMPBLAS_NAMED_TYPE(trans)
            << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(k)
            << AMPBLAS_NAMED_TYPE(alpha)
            << AMPBLAS_NAMED_TYPE(beta)
            << AMPBLAS_NAMED_TYPE(lda_offset)
            << AMPBLAS_NAMED_TYPE(ldb_offset)
            << AMPBLAS_NAMED_TYPE(ldc_offset);

        return out.str();
    }

};

template <typename value_type>
class syr2k_test : public test_case<value_type,syr2k_parameters>
{
public:

    std::string name() const
    {
        return "SYR2K";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        int nrowa;
        int ka;
	    if ( p.trans == AmpblasNoTrans )
	    {
	        nrowa = p.n;
	        ka = p.k;
	    }
	    else
	    {
	        nrowa = p.k;
	        ka = p.n;
	    }
        // derived parameters
        int lda = nrowa + p.lda_offset;
        int ldb = nrowa + p.ldb_offset;
        int ldc = p.n + p.ldc_offset;

        // reference data
        test_matrix<value_type> A(nrowa, ka, lda);
        test_matrix<value_type> B(nrowa, ka, ldb);
        test_matrix<value_type> C(p.n, p.n, ldc);

        // generate data
        randomize(A);
        randomize(B);
        randomize(C);

        // ampblas bound data
        ampblas_test_matrix<value_type> A_amp(A);
        ampblas_test_matrix<value_type> B_amp(B);
        ampblas_test_matrix<value_type> C_amp(C);

        // test references
        start_reference_test();
        cblas::xSYR2K(cblas_cast(p.uplo), cblas_cast(p.trans), p.n, p.k, cblas_cast(p.alpha), cblas_cast(A.data()), A.ld(), cblas_cast(B.data()), B.ld(), cblas_cast(p.beta), cblas_cast(C.data()), C.ld() );
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xsyr2k(AmpblasColMajor, p.uplo, p.trans, p.n, p.k, ampcblas_cast(p.alpha), ampcblas_cast(A_amp.data()), A_amp.ld(), ampcblas_cast(B_amp.data()), B_amp.ld(), ampcblas_cast(p.beta), ampcblas_cast(C_amp.data()), C_amp.ld() );
        stop_ampblas_test();

        // synchronize outputs
        C_amp.synchronize();

        // calculate error
        check_error(C, C_amp);
    }

    syr2k_test()
    {
        // bulk test
        std::vector<enum AMPBLAS_UPLO> uplo;
        uplo.push_back(AmpblasUpper);
        uplo.push_back(AmpblasLower);

        std::vector<enum AMPBLAS_TRANSPOSE> trans;
        trans.push_back(AmpblasNoTrans);
        trans.push_back(AmpblasTrans);
        trans.push_back(AmpblasConjTrans);

        std::vector<int> n;
        n.push_back(16);
        n.push_back(64);

        std::vector<int> k;
        k.push_back(16);
        k.push_back(64);

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

        paramter_exploder( uplo, trans, n, k, alpha, beta, lda_offset, ldb_offset, ldc_offset );
    }

    real_type fudge_factor() const 
    { 
        return real_type(50); 
    }
};

REGISTER_TEST(syr2k_test, float);
REGISTER_TEST(syr2k_test, double);
