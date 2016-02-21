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
 * scal_test.cpp
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
struct scal_parameters
{
    scal_parameters(int n, value_type alpha, int incx)
      : n(n), alpha(alpha), incx(incx)
    {}

    int n;
    value_type alpha;
    int incx;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(alpha) 
            << AMPBLAS_NAMED_TYPE(incx);

        return out.str();
    }

};

template <typename value_type>
class scal_test : public test_case<value_type,scal_parameters>
{
public:

    std::string name() const
    {
        return "SCAL";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // reference data
        test_vector<value_type> x(p.n, p.incx);

        // generate data
        randomize(x);

        // ampblas data
        ampblas_test_vector<value_type> x_amp(x);

        // test references
        start_reference_test();
        cblas::xSCAL(p.n, cblas_cast(p.alpha), cblas_cast(x.data()), x.inc());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xscal(p.n, ampcblas_cast(p.alpha), ampcblas_cast(x_amp.data()), x_amp.inc());
        stop_ampblas_test();

        // synchronize outputs
        x_amp.synchronize();

        // calculate error
        check_error(x, x_amp);
    }

    scal_test()
    {
        // bulk test example

        std::vector<int> n;
        n.push_back(16);
        n.push_back(64);
        n.push_back(256);
        n.push_back(2048);

        std::vector<value_type> alpha;
        alpha.push_back( value_type(1) );
        alpha.push_back( value_type(-1) );
        alpha.push_back( value_type(0) );

        std::vector<int> incx;
        incx.push_back(1);
        incx.push_back(-1);
        incx.push_back(2);

        paramter_exploder(n, alpha, incx);
    }
};

REGISTER_TEST(scal_test, float);
REGISTER_TEST(scal_test, double);
// REGISTER_TEST(scal_test, complex_float);
// REGISTER_TEST(scal_test, complex_double);
