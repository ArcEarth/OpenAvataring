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
 *---------------------------------------------------------------------------
 * 
 * Tests for AMP CBLAS interface ampblas_[s,d,c,z]swap
 *
 *---------------------------------------------------------------------------*/
#include <vector>
#include <complex>

#pragma warning (disable : 4244) // conversion from 'int' to 'const float', possible loss of data

//------------------------------------------------------------------------------------
// Testing ampblas_xswap: y[i*incy] = x[i*incx], 
// where incx > 0, incy > 0
//------------------------------------------------------------------------------------
template<typename T>
bool test_copy_1()
{
	const int n = 100;
	T x[n], y[n];

    // Initialize buffers 
	for (int i=0; i<n; i++)
	{
		x[i] = i;
		y[i] = 0;
	}

    // Bind buffers
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(x, n * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(y, n * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xcopy(n, x, 1, y, 1));
	EXECUTE_IF_OK(re, ampblas_synchronize(y, n * sizeof(T)));

    // Unbind buffers
	ampblas_unbind(x);
	ampblas_unbind(y);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<n; i++)
	{
		if (x[i] != y[i])
        {
            return false;
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xswap: y[i*incy] = x[i*incx], with complex type
// where incx > 0, incy > 0
//------------------------------------------------------------------------------------
template<typename T, typename compare_func>
bool test_copy_2(compare_func compare_equal)
{
	const int n = 100;
	T x[n], y[n];

    // Initialize buffers 
	for (int i=0; i<n; i++)
	{
        x[i].real = i; x[i].imag = i+1;
        y[i].real = 0; y[i].imag = 0;
	}

    // Bind buffers
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(x, n * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(y, n * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xcopy(n, x, 1, y, 1));
	EXECUTE_IF_OK(re, ampblas_synchronize(y, n * sizeof(T)));

    // Unbind buffers
	ampblas_unbind(x);
	ampblas_unbind(y);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<n; i++)
	{
		T actual = y[i];
        T expected = x[i];
		if (!compare_equal(&actual, &expected))
        {
            return false;
        }
	}

    return true;
}

