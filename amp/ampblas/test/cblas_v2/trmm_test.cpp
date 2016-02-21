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
 * trmm_text.cpp
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
struct trmm_parameters
{
    trmm_parameters(enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE trans, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, int lda_offset, int ldb_offset)
      : side(side), uplo(uplo), trans(trans), diag(diag), m(m), n(n), alpha(alpha), lda_offset(lda_offset), ldb_offset(ldb_offset)
    {}

    enum AMPBLAS_SIDE side;
    enum AMPBLAS_UPLO uplo;
    enum AMPBLAS_TRANSPOSE trans;
    enum AMPBLAS_DIAG diag;

    int m;
    int n;
    value_type alpha;
    int lda_offset;
    int ldb_offset;

    std::string name() const
    {
        std::stringstream out;

        char side = (this->side==AmpblasLeft)?'L':'R';
        char uplo = (this->uplo==AmpblasUpper)?'U':'L';
        char trans = (this->trans==AmpblasNoTrans)?'N':((this->trans==AmpblasTrans)?'T':'C');
        char diag = (this->diag==AmpblasUnit)?'U':'N';

        out << AMPBLAS_NAMED_TYPE(side)
            << AMPBLAS_NAMED_TYPE(uplo)
            << AMPBLAS_NAMED_TYPE(trans)
            << AMPBLAS_NAMED_TYPE(diag)
            << AMPBLAS_NAMED_TYPE(m)
            << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(alpha)
            << AMPBLAS_NAMED_TYPE(lda_offset)
            << AMPBLAS_NAMED_TYPE(ldb_offset);

        return out.str();
    }

};

template <typename value_type>
class trmm_test : public test_case<value_type,trmm_parameters>
{
public:

    std::string name() const
    {
        return "TRMM";
    }

    void run_cblas_test(const typed_parameters& p)
    {
	    int nrowa;
	    if ( p.side == AmpblasLeft )
	        nrowa = p.m;
	    else
	        nrowa = p.n;

        // derived parameters
        int lda = nrowa + p.lda_offset;
        int ldb = p.m + p.ldb_offset;

        // reference data
        test_matrix<value_type> A(nrowa, nrowa, lda);
        test_matrix<value_type> B(p.m, p.n, ldb);

        // generate data
        randomize(A);
        randomize(B);

        // ampblas bound data
        ampblas_test_matrix<value_type> A_amp(A);
        ampblas_test_matrix<value_type> B_amp(B);

        // test references
        start_reference_test();
        cblas::xTRMM(cblas_cast(p.side), cblas_cast(p.uplo), cblas_cast(p.trans), cblas_cast(p.diag), p.m, p.n, cblas_cast(p.alpha), cblas_cast(A.data()), A.ld(), cblas_cast(B.data()), B.ld());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xtrmm(AmpblasColMajor, p.side, p.uplo, p.trans, p.diag, p.m, p.n, ampcblas_cast(p.alpha), ampcblas_cast(A_amp.data()), A_amp.ld(), ampcblas_cast(B_amp.data()), B_amp.ld());
        stop_ampblas_test();

        // synchronize outputs
        B_amp.synchronize();

        // calculate error
        check_error(B, B_amp);
    }

    trmm_test()
    {
        // bulk test
        std::vector<enum AMPBLAS_SIDE> side;
        side.push_back(AmpblasLeft);
        side.push_back(AmpblasRight);

        std::vector<enum AMPBLAS_UPLO> uplo;
        uplo.push_back(AmpblasUpper);
        uplo.push_back(AmpblasLower);

        std::vector<enum AMPBLAS_TRANSPOSE> trans;
        trans.push_back(AmpblasNoTrans);
        trans.push_back(AmpblasTrans);
        trans.push_back(AmpblasConjTrans);

        std::vector<enum AMPBLAS_DIAG> diag;
        diag.push_back(AmpblasNonUnit);
        diag.push_back(AmpblasUnit);

        std::vector<int> m;
        m.push_back(16);
        m.push_back(20);
        m.push_back(64);

        std::vector<int> n;
        n.push_back(16);
        n.push_back(20);
        n.push_back(64);

        std::vector<value_type> alpha;
        alpha.push_back( value_type(1) );
        alpha.push_back( value_type(-1) );
        alpha.push_back( value_type(0) );

        std::vector<int> lda_offset;
        lda_offset.push_back(0);
        lda_offset.push_back(4);

        std::vector<int> ldb_offset;
        ldb_offset.push_back(0);
        ldb_offset.push_back(4);

        paramter_exploder( side, uplo, trans, diag, m, n, alpha, lda_offset, ldb_offset );
    }
};

REGISTER_TEST(trmm_test, float);
REGISTER_TEST(trmm_test, double);

// A potential compiler issue has rendered complex trmm incorrect
// REGISTER_TEST(trmm_test, complex_float);
// REGISTER_TEST(trmm_test, complex_double);
