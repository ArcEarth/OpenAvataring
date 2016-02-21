/*----------------------------------------------------------------------------
 * Copyright © Microsoft Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not 
 * use this file except in compliance with the License.  You may obtain a copy 
 * of the License at http://www.apache.org/licenses/LICENSE-2.0  
 * 
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED 
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, 
 * MERCHANTABLITY OR NON-INFRINGEMENT. 
 *
 * See the Apache Version 2.0 License for specific language governing 
 * permissions and limitations under the License.
 *---------------------------------------------------------------------------
 * 
 * syrk_text.cpp
 *
 *---------------------------------------------------------------------------*/

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
struct syrk_parameters
{
    typedef typename ampblas::real_type<value_type>::type real_type;

    syrk_parameters(enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, int n, int k, real_type alpha, real_type beta, int lda_offset, int ldc_offset)
      : uplo(uplo), trans(trans), n(n), k(k), alpha(alpha), beta(beta), lda_offset(lda_offset), ldc_offset(ldc_offset)
    {}

    enum AMPBLAS_UPLO uplo;
    enum AMPBLAS_TRANSPOSE trans;
    int n;
    int k;
    real_type alpha;
    real_type beta;
    int lda_offset;
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
            << AMPBLAS_NAMED_TYPE(ldc_offset);

        return out.str();
    }

};

template <typename value_type>
class syrk_test : public test_case<value_type,syrk_parameters>
{
public:

    typedef typename ampblas::real_type<value_type>::type real_type;

    std::string name() const
    {
        return "SYRK";
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
        int ldc = p.n + p.ldc_offset;

        // reference data
        test_matrix<value_type> A(nrowa, ka, lda);
        test_matrix<value_type> C(p.n, p.n, ldc);

        // generate data
        randomize(A);
        randomize(C);

        // ampblas bound data
        ampblas_test_matrix<value_type> A_amp(A);
        ampblas_test_matrix<value_type> C_amp(C);

        // test references
        start_reference_test();
        cblas::xSYRK(cblas_cast(p.uplo), cblas_cast(p.trans), p.n, p.k, p.alpha, cblas_cast(A.data()), A.ld(), p.beta, cblas_cast(C.data()), C.ld());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xsyrk(AmpblasColMajor, p.uplo, p.trans, p.n, p.k, p.alpha, ampcblas_cast(A_amp.data()), A_amp.ld(), p.beta, ampcblas_cast(C_amp.data()), C_amp.ld());
        stop_ampblas_test();

        // synchronize outputs
        C_amp.synchronize();

        // calculate error
        check_error(C, C_amp);
    }

    syrk_test()
    {
        // bulk test
        std::vector<enum AMPBLAS_UPLO> uplo;
        uplo.push_back(AmpblasUpper);
        uplo.push_back(AmpblasLower);

        std::vector<enum AMPBLAS_TRANSPOSE> trans;
        trans.push_back(AmpblasNoTrans);
        trans.push_back(AmpblasConjTrans);

        std::vector<int> n;
        n.push_back(16);
        n.push_back(19);
        n.push_back(64);

        std::vector<int> k;
        k.push_back(16);
        k.push_back(17);
        k.push_back(64);

        std::vector<real_type> alpha;
        alpha.push_back( real_type(1) );
        alpha.push_back( real_type(-1) );
        alpha.push_back( real_type(0) );

        std::vector<real_type> beta;
        beta.push_back( real_type(1) );
        beta.push_back( real_type(-1) );
        beta.push_back( real_type(0) );

        std::vector<int> lda_offset;
        lda_offset.push_back(0);
        lda_offset.push_back(4);

        std::vector<int> ldc_offset;
        ldc_offset.push_back(0);
        ldc_offset.push_back(4);

        paramter_exploder( uplo, trans, n, k, alpha, beta, lda_offset, ldc_offset );
    }
};

template <>
std::string syrk_test<complex_float>::name() const
{
    return "HERK";
}

template <>
std::string syrk_test<complex_double>::name() const
{
    return "HERK";
}

// syrk
REGISTER_TEST(syrk_test, float);
REGISTER_TEST(syrk_test, double);

// herk
REGISTER_TEST(syrk_test, complex_float);
REGISTER_TEST(syrk_test, complex_double);
