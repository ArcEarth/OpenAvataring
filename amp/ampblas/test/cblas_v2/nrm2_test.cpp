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
 * nrm2_text.cpp
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
struct nrm2_parameters
{
    nrm2_parameters(int n, int incx)
      : n(n), incx(incx)
    {}

    int n;
    int incx;

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(incx);

        return out.str();
    }

};

template <typename value_type>
class nrm2_test : public test_case<value_type,nrm2_parameters>
{
public:

    std::string name() const
    {
        return "NRM2";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // reference data
        ampblas_test_vector<value_type> x(p.n, p.incx);

        // generate data
        randomize(x);

        // test references
        start_reference_test();
        real_type cblas = cblas::xNRM2(x.n(), cblas_cast(x.data()), x.inc());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        real_type amp = ampblas_xnrm2(x.n(), ampcblas_cast(x.data()), x.inc());
        stop_ampblas_test();

        // calculate error
        check_error(x.n(), cblas, amp);
    }

    nrm2_test()
    {
        // Bulk Test Example

        std::vector<int> n;
        n.push_back(16);
        n.push_back(64);
        n.push_back(256);
        n.push_back(1024);

        std::vector<int> incx;
        incx.push_back(1);
        incx.push_back(-1);
        incx.push_back(2);

        paramter_exploder(n, incx);
    }
};

REGISTER_TEST(nrm2_test, float);
REGISTER_TEST(nrm2_test, double);
// REGISTER_TEST(nrm2_test, complex_float);
// REGISTER_TEST(nrm2_test, complex_double);
