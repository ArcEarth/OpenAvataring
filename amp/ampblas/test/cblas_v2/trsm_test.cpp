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
 * trsm_text.cpp
 *
 *---------------------------------------------------------------------------*/

// testing headers	
#include "ampblas_test_bench.h"
#include "ampblas_test_util.h"

#include <vector>
#include <sstream>

// unique paramaters for ger
template <typename value_type>
struct trsm_parameters
{
    trsm_parameters(enum AMPBLAS_SIDE side, enum AMPBLAS_UPLO uplo, enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_DIAG diag, int m, int n, value_type alpha, int lda_offset, int ldb_offset)
		: side(side), uplo(uplo), transa(transa), diag(diag), m(m), n(n), alpha(alpha), lda_offset(lda_offset), ldb_offset(ldb_offset)
	{}

	enum AMPBLAS_SIDE side;
	enum AMPBLAS_UPLO uplo;
	enum AMPBLAS_TRANSPOSE transa; 
	enum AMPBLAS_DIAG diag;
	int m;
	int n;
	value_type alpha; 
	int lda_offset;
	int ldb_offset;

    std::string name() const
    {
        std::stringstream out;

		out << AMPBLAS_NAMED_TYPE(side)
			<< AMPBLAS_NAMED_TYPE(uplo)
			<< AMPBLAS_NAMED_TYPE(transa) 
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
class trsm_test : public test_case<value_type,trsm_parameters>
{
public:

    std::string name() const
    {
        return "TRSM";
    }

    bool requires_full_double() const
    {
        // uses division
        return true;
    }

    void run_cblas_test(const typed_parameters& p)
    {
		// derived parameters
		int k = (p.side == AmpblasLeft ? p.m : p.n);

		// column major
		int lda = k + p.lda_offset;
		int ldb = p.m + p.ldb_offset;

        // reference data
        ampblas_test_matrix<value_type> A(k, k, lda);
        test_matrix<value_type> B(p.m, p.n, ldb);     

        // generate data
        randomize(A, value_type(1), value_type(2));
        randomize(B);

        // this will quickly generate a "happy" matrix that can be solved without floating point overflow
        for (int i=0; i<k; i++)
        {
            value_type Aii = A(i,i);
            A(i,i) = value_type(1);
            real_type scale = cblas::xNRM2(k, cblas_cast(A.data()+i*A.ld()), 1);
            cblas::xSCAL(k, real_type(1)/scale, cblas_cast(A.data()+i*A.ld()), 1);
            A(i,i) = Aii;
        }
        
        // ampblas data
        ampblas_test_matrix<value_type> B_amp(B);     

        // test references
        start_reference_test();
		cblas::xTRSM(cblas_cast(p.side), cblas_cast(p.uplo), cblas_cast(p.transa), cblas_cast(p.diag), p.m, p.n, cblas_cast(p.alpha), cblas_cast(A.data()), A.ld(), cblas_cast(B.data()), B.ld());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
		ampblas_xtrsm(AmpblasColMajor, p.side, p.uplo, p.transa, p.diag, p.m, p.n, ampcblas_cast(p.alpha), ampcblas_cast(A.data()), A.ld(), ampcblas_cast(B_amp.data()), B_amp.ld());
        stop_ampblas_test();

        // synchronize outputs
        B_amp.synchronize();

        // calculate error
        check_error(B, B_amp);
    }

    trsm_test()
    {
        // bulk test example
		std::vector<enum AMPBLAS_SIDE> side;
		side.push_back(AmpblasLeft);
		side.push_back(AmpblasRight);

		std::vector<enum AMPBLAS_UPLO> uplo;
		uplo.push_back(AmpblasUpper);
		uplo.push_back(AmpblasLower);

		std::vector<enum AMPBLAS_TRANSPOSE> transa;
		transa.push_back(AmpblasNoTrans);
		transa.push_back(AmpblasTrans);
        transa.push_back(AmpblasConjTrans);

		std::vector<enum AMPBLAS_DIAG> diag;
		diag.push_back(AmpblasNonUnit);
		diag.push_back(AmpblasUnit);

		std::vector<int> m;
        m.push_back(64);
        m.push_back(96);
        
        std::vector<int> n;
        n.push_back(64);
        n.push_back(96);

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

        paramter_exploder(side,uplo,transa,diag,m,n,alpha,lda_offset,ldb_offset);
    }
};

REGISTER_TEST(trsm_test, float);
REGISTER_TEST(trsm_test, double);
REGISTER_TEST(trsm_test, complex_float);
REGISTER_TEST(trsm_test, complex_double);
