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
 * gemm_text.cpp
 *
 *---------------------------------------------------------------------------*/

// testing headers
#include "ampblas_test_bench.h"
#include "ampblas_test_util.h"

// routine details
#include "detail/gemm.h"

#include <vector>
#include <sstream>

// unique paramaters for ger
template <typename value_type>
struct gemm_parameters
{
    gemm_parameters(enum AMPBLAS_TRANSPOSE transa, enum AMPBLAS_TRANSPOSE transb, int m, int n, int k, value_type alpha, value_type beta, int lda_offset, int ldb_offset, int ldc_offset)
      : transa(transa), transb(transb), m(m), n(n), k(k), alpha(alpha), beta(beta), lda_offset(lda_offset), ldb_offset(ldb_offset), ldc_offset(ldc_offset)
    {}

	enum AMPBLAS_TRANSPOSE transa;
	enum AMPBLAS_TRANSPOSE transb;
	int m;
    int n;
	int k;
    value_type alpha;
	value_type beta;
    int lda_offset;
    int ldb_offset;
	int ldc_offset;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(transa)
			<< AMPBLAS_NAMED_TYPE(transb)
			<< AMPBLAS_NAMED_TYPE(m)
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
class gemm_test : public test_case<value_type,gemm_parameters>
{
public:

    std::string name() const
    {
        return "GEMM";
    }

    double flops(const typed_parameters& p)
    {
        return 2.0 * double(p.m) * double(p.n) * double(p.k);
    }

    void run_cblas_test(const typed_parameters& p)
    {
		// derived parameters
		auto row_a = (p.transa == AmpblasNoTrans ? p.m : p.k);
		auto col_a = (p.transa == AmpblasNoTrans ? p.k : p.m);
		auto row_b = (p.transb == AmpblasNoTrans ? p.k : p.n);
		auto col_b = (p.transb == AmpblasNoTrans ? p.n : p.k);

		// column major
		int lda = row_a + p.lda_offset;
		int ldb = row_b + p.ldb_offset;
		int ldc = p.m + p.ldc_offset;

        // reference data
        ampblas_test_matrix<value_type> A(row_a, col_a, lda);
        ampblas_test_matrix<value_type> B(row_b, col_b, ldb);     
		test_matrix<value_type> C(p.m, p.n, ldc);

        // generate data
        randomize(A);
        randomize(B);
		randomize(C);

        // ampblas data
        ampblas_test_matrix<value_type> C_amp(C);

        // test references
        start_reference_test();
		cblas::xGEMM(cblas_cast(p.transa), cblas_cast(p.transb), p.m, p.n, p.k, cblas_cast(p.alpha), cblas_cast(A.data()), A.ld(), cblas_cast(B.data()), B.ld(), cblas_cast(p.beta), cblas_cast(C.data()), C.ld());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
		ampblas_xgemm(AmpblasColMajor, p.transa, p.transb, p.m, p.n, p.k, ampcblas_cast(p.alpha), ampcblas_cast(A.data()), A.ld(), ampcblas_cast(B.data()), B.ld(), ampcblas_cast(p.beta), ampcblas_cast(C_amp.data()), C_amp.ld());
        stop_ampblas_test();

        // synchronize outputs
        C_amp.synchronize();

        // calculate error
        check_error(C, C_amp);
    }

    gemm_test()
    {
        // bulk test example
		std::vector<enum AMPBLAS_TRANSPOSE> transa;
		transa.push_back(AmpblasNoTrans);
		transa.push_back(AmpblasTrans);
        transa.push_back(AmpblasConjTrans);

		std::vector<enum AMPBLAS_TRANSPOSE> transb;
		transb.push_back(AmpblasNoTrans);
		transb.push_back(AmpblasTrans);
        transb.push_back(AmpblasConjTrans);

		std::vector<int> m;
        m.push_back(20);
        m.push_back(128);

        std::vector<int> n;
        n.push_back(20);
        n.push_back(128);

		std::vector<int> k;
        k.push_back(20);
        k.push_back(128);

        std::vector<value_type> alpha;
        alpha.push_back( value_type(1) );
        alpha.push_back( value_type(-1) );
        alpha.push_back( value_type(0) );

		std::vector<value_type> beta;
        beta.push_back( value_type(1) );
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

        paramter_exploder(transa,transb,m,n,k,alpha,beta,lda_offset,ldb_offset,ldc_offset);
    }
};

REGISTER_TEST(gemm_test, float);
REGISTER_TEST(gemm_test, double);
REGISTER_TEST(gemm_test, complex_float);
REGISTER_TEST(gemm_test, complex_double);
