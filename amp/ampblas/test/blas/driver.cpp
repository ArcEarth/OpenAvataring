/* 
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
 */

#include <iostream>
#include <assert.h>
#include "ampblas.h"

#pragma warning (disable : 4244) // conversion from 'int' to 'const float', possible loss of data

typedef ampblas::complex<float>  fcomplex;
typedef ampblas::complex<double> dcomplex;

//------------------------------------------------------------------------------------
// Testing ampblas::axpy with real type
//------------------------------------------------------------------------------------
template<typename T>
bool test_axpy_1()
{
	const int n = 100;
	T x[n], y[n], alpha = 17;

    for (int i=0; i<n; i++)
	{
		x[i] = (T)i;
		y[i] = (T)i * 10;
	}

    try 
    {
	    ampblas::bind(x, n);
	    ampblas::bind(y, n);

	    ampblas::axpy(n, alpha, x, 1, y, 1);
	    ampblas::synchronize(y, n);

	    ampblas::unbind(x);
	    ampblas::unbind(y);

	    for (int i=0; i<n; i++)
	    {
		    T actual = y[i];
		    T expected = static_cast<T>(i * (10 + 17));
		    if (actual != expected)
            {
                return false;
            }
	    }
    }
    catch (std::exception&)
    {
        // unbind buffers
        ampblas::unbind(x);
        ampblas::unbind(y);

        return false;
    }

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas::axpy with complex type
//------------------------------------------------------------------------------------
template<typename T>
bool test_axpy_2()
{
	const int n = 100;
	T x[n], y[n], alpha = T(17,12);

    for (int i=0; i<n; i++)
	{
		x[i] = T(i, i);
		y[i] = T(i, i) * T::value_type(10);
	}

    try 
    {
	    ampblas::bind(x, n);
	    ampblas::bind(y, n);

	    ampblas::axpy(n, alpha, x, 1, y, 1);
	    ampblas::synchronize(y, n);

	    ampblas::unbind(x);
	    ampblas::unbind(y);

	    for (int i=0; i<n; i++)
	    {
		    T actual = y[i];
		    T expected = T(15*i, 39*i);
		    if (actual != expected)
            {
                return false;
            }
	    }
    }
    catch (std::exception&)
    {
        // unbind buffers
        ampblas::unbind(x);
        ampblas::unbind(y);

        return false;
    }

    return true;
}

inline bool run_test(bool test, const char* test_name)
{
    bool result = test;

    std::cout << test_name << (result ? ": passed\n" : ": failed\n");

    return result;
}

int main()
{
    bool passed = true;

    // Testing axpy
    passed &= run_test(test_axpy_1<float>(), "test_axpy_1<float>");
    passed &= run_test(test_axpy_1<double>(), "test_axpy_1<double>");
	passed &= run_test(test_axpy_2<fcomplex>(), "test_axpy_2<fcomplex>");
	passed &= run_test(test_axpy_2<dcomplex>(), "test_axpy_2<dcomplex>");

    return !passed;
}
