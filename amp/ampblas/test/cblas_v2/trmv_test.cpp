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
 * trmv_text.cpp
 *
 *---------------------------------------------------------------------------*/

// testing headers
#include "ampblas_test_bench.h"
#include "ampblas_test_util.h"

#include <vector>
#include <sstream>

// unique paramaters for ger
template <typename value_type>
struct trmv_parameters
{
    trmv_parameters(enum AMPBLAS_UPLO uplo, AMPBLAS_TRANSPOSE trans, AMPBLAS_DIAG diag, int n, int lda_offset, int incx)
      : uplo(uplo), trans(trans), diag(diag), n(n), lda_offset(lda_offset), incx(incx)
    {}

	enum AMPBLAS_UPLO uplo;
    enum AMPBLAS_TRANSPOSE trans;
    enum AMPBLAS_DIAG diag;
    int n;
    int lda_offset;
    int incx;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(uplo)
            << AMPBLAS_NAMED_TYPE(trans)
            << AMPBLAS_NAMED_TYPE(diag)
            << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(lda_offset)
            << AMPBLAS_NAMED_TYPE(incx);

        return out.str();
    }
};

template <typename value_type>
class trmv_test : public test_case<value_type,trmv_parameters>
{
public:

    std::string name() const
    {
        return "TRMV";
    }

    real_type fudge_factor() const
    {
        // n^2 operation
        return real_type(30);
    }

    void run_cblas_test(const typed_parameters& p)
    {
		// column major
		int lda = p.n + p.lda_offset;

        // reference data
        ampblas_test_matrix<value_type> A(p.n, p.n, lda);
		test_vector<value_type> x(p.n, p.incx);

        // generate data
        randomize(A);
        randomize(x);

        // ampblas data
        ampblas_test_vector<value_type> x_amp(x);

        // test references
        start_reference_test();
		cblas::xTRMV(cblas_cast(p.uplo), cblas_cast(p.trans), cblas_cast(p.diag), p.n, cblas_cast(A.data()), A.ld(), cblas_cast(x.data()), x.inc());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
		ampblas_xtrmv(AmpblasColMajor, p.uplo, p.trans, p.diag, p.n, ampcblas_cast(A.data()), A.ld(), ampcblas_cast(x_amp.data()), x_amp.inc());
        stop_ampblas_test();

        // synchronize outputs
        x_amp.synchronize();

        // calculate error
        check_error(x, x_amp);
    }

    trmv_test()
    {
        // bulk test example
		std::vector<enum AMPBLAS_UPLO> uplo;
		uplo.push_back(AmpblasUpper);
		uplo.push_back(AmpblasLower);

		std::vector<enum AMPBLAS_TRANSPOSE> transa;
		transa.push_back(AmpblasNoTrans);
		transa.push_back(AmpblasTrans);

		std::vector<enum AMPBLAS_DIAG> diag;
		diag.push_back(AmpblasNonUnit);
		diag.push_back(AmpblasUnit);

        std::vector<int> n;
        n.push_back(16);
        n.push_back(256);

		std::vector<int> lda_offset;
		lda_offset.push_back(0);
		//lda_offset.push_back(4);

		std::vector<int> incx;
		incx.push_back(1);
		//incx.push_back(-1);
		//incx.push_back(2);

        paramter_exploder(uplo,transa,diag,n,lda_offset,incx);
    }
};

REGISTER_TEST(trmv_test, float)
REGISTER_TEST(trmv_test, complex_float);

// runtime error in release mode
REGISTER_TEST(trmv_test, double);
REGISTER_TEST(trmv_test, complex_double);
