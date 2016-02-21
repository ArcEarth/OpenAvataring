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
 * syr_text.cpp
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
struct syr_parameters
{
    syr_parameters(enum AMPBLAS_UPLO uplo, int n, typename ampblas::real_type<value_type>::type alpha, int incx, int lda_offset)
      : uplo(uplo), n(n), alpha(alpha), incx(incx), lda_offset(lda_offset)
    {}

    enum AMPBLAS_UPLO uplo;
    int n;
    typename ampblas::real_type<value_type>::type alpha;
    int incx;
    int lda_offset;

    std::string name() const
    {
        std::stringstream out;
 
        char uplo = (this->uplo==AmpblasUpper) ? 'U' : 'L';

        out << AMPBLAS_NAMED_TYPE(uplo)
            << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(alpha)
            << AMPBLAS_NAMED_TYPE(incx)
            << AMPBLAS_NAMED_TYPE(lda_offset);

        return out.str();
    }

};

template <typename value_type>
class syr_test : public test_case<value_type,syr_parameters>
{
public:

    std::string name() const
    {
        return "SYR/HER";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // derived parameters
        int lda = p.n + p.lda_offset;

        // reference data
        ampblas_test_vector<value_type> x(p.n, p.incx);
        test_matrix<value_type> A(p.n, p.n, lda);

        // generate data
        randomize(x);
        randomize(A);

        // ampblas bound data
        ampblas_test_matrix<value_type> A_amp(A);

        // test references
        start_reference_test();
        cblas::xSYR(cblas_cast(p.uplo), p.n, p.alpha, cblas_cast(x.data()), x.inc(), cblas_cast(A.data()), A.ld());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xsyr(AmpblasColMajor, p.uplo, p.n, p.alpha, ampcblas_cast(x.data()), x.inc(), ampcblas_cast(A_amp.data()), A_amp.ld());
        stop_ampblas_test();

        // synchronize outputs
        A_amp.synchronize();

        // calculate error
        check_error(A, A_amp);
    }

    syr_test()
    {
        // bulk test
        std::vector<enum AMPBLAS_UPLO> uplo;
        uplo.push_back(AmpblasUpper);
        uplo.push_back(AmpblasLower);

        std::vector<int> n;
        n.push_back(1);
        n.push_back(16);
        //n.push_back(64);

        typedef typename ampblas::real_type<value_type>::type alpha_type;
        std::vector<alpha_type> alpha;
        alpha.push_back( alpha_type(1) );
        //alpha.push_back( alpha_type(-1) );
        //alpha.push_back( alpha_type(0) );

        std::vector<int> incx;
        incx.push_back(1);
        //incx.push_back(-1);
        //incx.push_back(2);

        std::vector<int> lda_offset;
        lda_offset.push_back(0);
        //lda_offset.push_back(4);

        paramter_exploder(uplo, n, alpha, incx, lda_offset);
    }
};

// REGISTER_TEST(syr_test, float);
// REGISTER_TEST(syr_test, double);

// her
REGISTER_TEST(syr_test, complex_float);
REGISTER_TEST(syr_test, complex_double);
