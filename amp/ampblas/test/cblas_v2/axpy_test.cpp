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
 * axpy_test.cpp
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
struct axpy_parameters
{
    axpy_parameters(int n, value_type alpha, int incx, int incy)
      : n(n), alpha(alpha), incx(incx), incy(incy)
    {}

    int n;
    value_type alpha;
    int incx;
    int incy; 

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(alpha) 
            << AMPBLAS_NAMED_TYPE(incx) 
            << AMPBLAS_NAMED_TYPE(incy);

        return out.str();
    }

};

template <typename value_type>
class axpy_test : public test_case<value_type,axpy_parameters>
{
public:

    std::string name() const
    {
        return "AXPY";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // reference data
        ampblas_test_vector<value_type> x(p.n, p.incx);
        test_vector<value_type> y(p.n, p.incy);     

        // generate data
        randomize(x);
        randomize(y);

        // ampblas data
        ampblas_test_vector<value_type> y_amp(y);

        // test references
        start_reference_test();
        cblas::xAXPY( p.n, cblas_cast(p.alpha), cblas_cast(x.data()), x.inc(), cblas_cast(y.data()), y.inc() );
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampblas_xaxpy( p.n, ampcblas_cast(p.alpha), ampcblas_cast(x.data()), x.inc(), ampcblas_cast(y_amp.data()), y_amp.inc() );
        stop_ampblas_test();

        // synchronize outputs
        y_amp.synchronize();

        // calculate error
        check_error(y, y_amp);
    }

    axpy_test()
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

        std::vector<int> incy;
        incy.push_back(1);
        incy.push_back(-1);
        incy.push_back(2);

        paramter_exploder(n, alpha, incx, incy);
    }
};

REGISTER_TEST(axpy_test, float);
REGISTER_TEST(axpy_test, double);
REGISTER_TEST(axpy_test, complex_float);
REGISTER_TEST(axpy_test, complex_double);
