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
 * gemv_text.cpp
 *
 *---------------------------------------------------------------------------*/

// testing headers
#include "ampblas_test_bench.h"
#include "ampblas_test_util.h"

#include <vector>
#include <sstream>

// unique paramaters for ger
template <typename value_type>
struct gemv_parameters
{
    gemv_parameters(enum AMPBLAS_TRANSPOSE transa, int m, int n, value_type alpha, int lda_offset, int incx, value_type beta, int incy)
      : transa(transa), m(m), n(n), alpha(alpha), lda_offset(lda_offset), incx(incx), beta(beta), incy(incy)
    {}

	enum AMPBLAS_TRANSPOSE transa;
	int m;
    int n;
    value_type alpha;
    int lda_offset;
	value_type beta;
    int incx;
	int incy;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(transa)
			<< AMPBLAS_NAMED_TYPE(m)
            << AMPBLAS_NAMED_TYPE(n) 
            << AMPBLAS_NAMED_TYPE(alpha)
			<< AMPBLAS_NAMED_TYPE(lda_offset)
			<< AMPBLAS_NAMED_TYPE(incx)
			<< AMPBLAS_NAMED_TYPE(beta)
			<< AMPBLAS_NAMED_TYPE(incy);

        return out.str();
    }

};

template <typename value_type>
class gemv_test : public test_case<value_type,gemv_parameters>
{
public:

    std::string name() const
    {
        return "GEMV";
    }

    void run_cblas_test(const typed_parameters& p)
    {
		int lda = p.m + p.lda_offset;

        // reference data
        ampblas_test_matrix<value_type> A(p.m, p.n, lda);
		ampblas_test_vector<value_type> x((p.transa == AmpblasNoTrans ? p.n : p.m), p.incx);
		test_vector<value_type> y((p.transa == AmpblasNoTrans ? p.m : p.n), p.incy);

        // generate data
        randomize(A);
        randomize(x);
		randomize(y);

        // ampblas data
        ampblas_test_vector<value_type> y_amp(y);

        // test references
        start_reference_test();
		cblas::xGEMV(cblas_cast(p.transa), p.m, p.n, cblas_cast(p.alpha), cblas_cast(A.data()), A.ld(), cblas_cast(x.data()), x.inc(), cblas_cast(p.beta), cblas_cast(y.data()), y.inc());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
		ampblas_xgemv(AmpblasColMajor, p.transa, p.m, p.n, ampcblas_cast(p.alpha), ampcblas_cast(A.data()), A.ld(), ampcblas_cast(x.data()), x.inc(), ampcblas_cast(p.beta), ampcblas_cast(y_amp.data()), y_amp.inc());
        stop_ampblas_test();

        // synchronize outputs
        y_amp.synchronize();

        // calculate error
        check_error(y, y_amp);
    }

    gemv_test()
    {
        // bulk test example
		std::vector<enum AMPBLAS_TRANSPOSE> transa;
		transa.push_back(AmpblasNoTrans);
		transa.push_back(AmpblasTrans);
        transa.push_back(AmpblasConjTrans);

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

		std::vector<int> lda_offset;
		lda_offset.push_back(0);
		lda_offset.push_back(4);

		std::vector<int> incx;
		incx.push_back(1);
		incx.push_back(-1);
		incx.push_back(2);

		std::vector<value_type> beta;
        beta.push_back( value_type(1) );
        beta.push_back( value_type(-1) );
        beta.push_back( value_type(0) );
		
		std::vector<int> incy;
		incy.push_back(1);
		incy.push_back(-1);
		incy.push_back(2);

        paramter_exploder(transa,m,n,alpha,lda_offset,incx,beta,incy);
    }
};

REGISTER_TEST(gemv_test, float);
REGISTER_TEST(gemv_test, double);
REGISTER_TEST(gemv_test, complex_float);
REGISTER_TEST(gemv_test, complex_double);
