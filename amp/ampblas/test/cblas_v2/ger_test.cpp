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
 * ger_text.cpp
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
struct ger_parameters
{
    ger_parameters(int m, int n, value_type alpha, int incx, int incy, int lda_offset)
      : m(m), n(n), alpha(alpha), incx(incx), incy(incy), lda_offset(lda_offset)
    {}

    int m;
    int n;
    value_type alpha;
    int incx;
    int incy; 
    int lda_offset;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(m) 
            << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(alpha) 
            << AMPBLAS_NAMED_TYPE(incx) 
            << AMPBLAS_NAMED_TYPE(incy) 
            << AMPBLAS_NAMED_TYPE(lda_offset);

        return out.str();
    }

};

template <typename value_type>
class ger_test : public test_case<value_type,ger_parameters>
{
public:

    std::string name() const
    {
        return "GER";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // derived parameters
        int lda = p.m + p.lda_offset;

        // reference data
        ampblas_test_vector<value_type> x(p.m, p.incx); 
        ampblas_test_vector<value_type> y(p.n, p.incy); 
        test_matrix<value_type> A(p.m, p.n, lda);

        // generate data
        randomize(x);
        randomize(y);
        randomize(A);

        // ampblas bound data
        ampblas_test_matrix<value_type> A_amp(A);

        // test references
        start_reference_test();
        cblas::xGER(p.m, p.n, cblas_cast(p.alpha), cblas_cast(x.data()), x.inc(), cblas_cast(y.data()), y.inc(), cblas_cast(A.data()), A.ld());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xger(AmpblasColMajor, p.m, p.n, ampcblas_cast(p.alpha), ampcblas_cast(x.data()), x.inc(), ampcblas_cast(y.data()), y.inc(), ampcblas_cast(A_amp.data()), A_amp.ld());
        stop_ampblas_test();

        // synchronize outputs
        A_amp.synchronize();

        // calculate error
        check_error(A, A_amp);
    }

    ger_test()
    {
        // bulk test
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

        std::vector<int> incx;
        incx.push_back(1);
        incx.push_back(-1);
        incx.push_back(2);

        std::vector<int> incy;
        incy.push_back(1);
        incy.push_back(-1);
        incy.push_back(2);

        std::vector<int> lda_offset;
        lda_offset.push_back(0);
        lda_offset.push_back(4);

        paramter_exploder( m, n, alpha, incx, incy, lda_offset );
    }
};

REGISTER_TEST(ger_test, float);
REGISTER_TEST(ger_test, double);
REGISTER_TEST(ger_test, complex_float);
REGISTER_TEST(ger_test, complex_double);
