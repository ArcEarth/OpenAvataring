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
 * dot_text.cpp
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
struct dot_parameters
{
    dot_parameters(int n, int incx, int incy)
      : n(n), incx(incx), incy(incy)
    {}

    int n;
    int incx;
    int incy; 

    std::string name() const
    {
        std::stringstream out;

        out << AMPBLAS_NAMED_TYPE(n)
            << AMPBLAS_NAMED_TYPE(incx) 
            << AMPBLAS_NAMED_TYPE(incy);

        return out.str();
    }

};

template <typename value_type>
class dot_test : public test_case<value_type, dot_parameters>
{
public:

    std::string name() const
    {
        return "DOT";
    }

    void run_cblas_test(const typed_parameters& p)
    {
        // input data
        ampblas_test_vector<value_type> x(p.n, p.incx);
        ampblas_test_vector<value_type> y(p.n, p.incy);     

        // generate data
        randomize(x);
        randomize(y);

        typedef typename cblas_type<value_type>::type cblas_type;
        typedef typename ampcblas_type<value_type>::type ampcblas_type;

        // test references
        start_reference_test();
        cblas_type cblas = cblas::xDOT<cblas_type,cblas_type>(p.n, cblas_cast(x.data()), x.inc(), cblas_cast(y.data()), y.inc());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        ampcblas_type amp = ampblas_xdot(p.n, ampcblas_cast(x.data()), x.inc(), ampcblas_cast(y.data()), y.inc());
        stop_ampblas_test();

        // calculate error
        value_type cblas_val = *reinterpret_cast<value_type*>(&cblas);
        value_type amp_val   = *reinterpret_cast<value_type*>(&amp);
        check_error(p.n, cblas_val, amp_val);
    }

    dot_test()
    {
        // bulk test example

        std::vector<int> n;
        n.push_back(16);
        n.push_back(64);
        n.push_back(256);
        n.push_back(2048);

        std::vector<int> incx;
        incx.push_back(1);
        incx.push_back(-1);
        incx.push_back(2);

        std::vector<int> incy;
        incy.push_back(1);
        incy.push_back(-1);
        incy.push_back(2);

        paramter_exploder(n, incx, incy);
    }
};

template <typename value_type>
class promoted_dot_test : public dot_test<value_type>
{
    std::string name() const
    {
        return "DOT+";
    }

	bool is_double() const
	{
		// need to check promoted type 
		typedef typename get_real_type<promoted_type>::value real_promoted_type;
		return ( typeid(real_promoted_type) == typeid(double) );
	}

    void run_cblas_test(const typed_parameters& p)
    {
        // input data
        ampblas_test_vector<value_type> x(p.n, p.incx);
        ampblas_test_vector<value_type> y(p.n, p.incy);     

        // generate data
        randomize(x);
        randomize(y);

        // test references
        start_reference_test();
        promoted_type cblas = cblas::xDOT<value_type,promoted_type>(p.n, cblas_cast(x.data()), x.inc(), cblas_cast(y.data()), y.inc());
        stop_reference_test();

        // test ampblas
        start_ampblas_test();
        promoted_type amp = ampblas_dsdot(p.n, ampcblas_cast(x.data()), x.inc(), ampcblas_cast(y.data()), y.inc());
        stop_ampblas_test();

        // calculate error
        check_error_promoted(p.n, cblas, amp);
    }
};

REGISTER_TEST(dot_test, float);
REGISTER_TEST(dot_test, double);
REGISTER_TEST(dot_test, complex_float);
REGISTER_TEST(dot_test, complex_double);

REGISTER_TEST(promoted_dot_test, float);
