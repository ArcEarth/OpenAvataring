/* 
 * Copyright © Microsoft Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not 
 * use this file except in compliance with the License.  You may obtain a scal 
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
 * Tests for AMP CBLAS interface ampblas_[s,d,c,z]scal
 *
 *---------------------------------------------------------------------------*/
#include <vector>
#include <complex>

#pragma warning (disable : 4244) // conversion from 'int' to 'const float', possible loss of data

//------------------------------------------------------------------------------------
// Testing ampblas_xscal: x[i*incx] *= alpha, 
// where incx > 0, incy > 0
//------------------------------------------------------------------------------------
template<typename T>
bool test_scal_1()
{
	const int n = 100;
	T x[n], alpha = 17;

    // Initialize buffers 
	for (int i=0; i<n; i++)
	{
		x[i] = i;
	}

    // Bind buffer
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(x, n * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xscal(n, alpha, x, 1));
	EXECUTE_IF_OK(re, ampblas_synchronize(x, n * sizeof(T)));

    // Unbind buffer
	ampblas_unbind(x);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<n; i++)
	{
		if (x[i] != alpha * i)
        {
            return false;
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xscal: y[i*incy] += alpha * x[i*incx], with complex type
// where incx > 0, incy > 0
//------------------------------------------------------------------------------------
template<typename T, typename compare_func>
bool test_scal_2(compare_func compare_equal)
{
	const int n = 100;
    T x[n], alpha = {3,2};

    // Initialize buffer 
	for (int i=0; i<n; i++)
	{
        x[i].real = i; x[i].imag = i;
	}

    // Bind buffer
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(x, n * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xscal(n, alpha, x, 1));
	EXECUTE_IF_OK(re, ampblas_synchronize(x, n * sizeof(T)));

    // Unbind buffer
	ampblas_unbind(x);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<n; i++)
	{
		T actual = x[i];
        T expected = {i, 5*i};
		if (!compare_equal(&actual, &expected))
        {
            return false;
        }
	}

    return true;
}

