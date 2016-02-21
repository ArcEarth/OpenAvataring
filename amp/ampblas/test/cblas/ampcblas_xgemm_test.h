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
 * Tests for AMP CBLAS interface ampblas_[s,d,c,z]gemm
 *
 *---------------------------------------------------------------------------*/
#include <vector>
#include <complex>

#pragma warning (disable : 4244) // conversion from 'int' to 'const float', possible loss of data

//------------------------------------------------------------------------------------
// Testing ampblas_xgemm: c = alpha * a * b + beta * c, 
//   lda == ldb == ldc == m == n == k
//------------------------------------------------------------------------------------
template<typename T>
bool test_gemm_1()
{
	const int m = 64;
    const int n = m;
    const int k = m;
    const int lda = k, ldb = n, ldc = n;
	T a[m*lda], b[k*ldb], c[m*ldc], alpha = 2, beta = 3;

    // Initialize buffers 
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            a[i*lda+j] = 2;
            b[i*ldb+j] = 0;
            c[i*ldc+j] = 3;
            if (i == j)
            {
                b[i*ldb+j] = j;
            }
        }
	}

    // Bind buffers
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(a, m * lda * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(b, k * ldb * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(c, m * ldc * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xgemm(AmpblasRowMajor, AmpblasNoTrans, AmpblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
	EXECUTE_IF_OK(re, ampblas_synchronize(c, m * ldc * sizeof(T)));

    // Unbind buffers
	ampblas_unbind(a);
	ampblas_unbind(b);
	ampblas_unbind(c);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            auto expected = (2 *j * alpha + 3 * beta);
            auto actual  = c[i*ldc+j];

			if (fabs(actual - expected) / fabs(actual + expected) > 0.0001) 
			{
				return false;
			}
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xgemm: c = alpha * a * b + beta * c, 
//   with complex type and lda == ldb == ldc == m == n == k
//------------------------------------------------------------------------------------
template<typename T, typename compare_func>
bool test_gemm_2(compare_func compare_equal)
{
	const int m = 64;
    const int n = m;
    const int k = m;
    const int lda = k, ldb = n, ldc = n;
    T a[m*lda], b[k*ldb], c[m*ldc], alpha = {2,1}, beta = {3,2};

    // Initialize buffers 
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            a[i*lda+j].real = 2; a[i*lda+j].imag = 1;
            b[i*ldb+j].real = 0; b[i*ldb+j].imag = 0;
            c[i*ldc+j].real = 1; c[i*ldc+j].imag = 2;
            if (i == j)
            {
                b[i*ldb+j].real = j; b[i*ldb+j].imag = j;
            }
        }
	}

    // Bind buffers
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(a, m * lda * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(b, k * ldb * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(c, m * ldc * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xgemm(AmpblasRowMajor, AmpblasNoTrans, AmpblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
	EXECUTE_IF_OK(re, ampblas_synchronize(c, m * ldc * sizeof(T)));

    // Unbind buffers
	ampblas_unbind(a);
	ampblas_unbind(b);
	ampblas_unbind(c);


    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            T expected = {-j-1,7*j+8}; // (2,1) *(j,3*j) + (3,2)*(1,2)
            T actual  = c[i*ldc+j];

            if (!compare_equal(&actual, &expected))
            {
                return false;
            }
        }
	}



    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xgemm: c = alpha * a * b + beta * c, 
//   lda != k, ldb != n, ldc != n
//------------------------------------------------------------------------------------
template<typename T>
bool test_gemm_3()
{
	const int m = 64;
    const int n = m;
    const int k = m;
    const int lda = k+32, ldb = n+32, ldc = n+32;
	T a[m*lda], b[k*ldb], c[m*ldc], alpha = 2, beta = 3;

    // Initialize buffers 
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            a[i*lda+j] = 2;
            b[i*ldb+j] = 0;
            c[i*ldc+j] = 3;
            if (i == j)
            {
                b[i*ldb+j] = j;
            }
        }
	}

    // Bind buffers
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(a, m * lda * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(b, k * ldb * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(c, m * ldc * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xgemm(AmpblasRowMajor, AmpblasNoTrans, AmpblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
	EXECUTE_IF_OK(re, ampblas_synchronize(c, m * ldc * sizeof(T)));

    // Unbind buffers
	ampblas_unbind(a);
	ampblas_unbind(b);
	ampblas_unbind(c);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            auto expected = (2 *j * alpha + 3 * beta);
            auto actual  = c[i*ldc+j];

			if (fabs(actual - expected) / fabs(actual + expected) > 0.0001) 
			{
				return false;
			}
        }
	}

    return true;
}

//------------------------------------------------------------------------------------
// Testing ampblas_xgemm: c = alpha * a * b + beta * c, 
//   with complex type and lda != m, ldb != n, ldc != n
//------------------------------------------------------------------------------------
template<typename T, typename compare_func>
bool test_gemm_4(compare_func compare_equal)
{
	const int m = 64;
    const int n = m;
    const int k = m;
    const int lda = k+32, ldb = n+24, ldc = n+16;
    T a[m*lda], b[k*ldb], c[m*ldc], alpha = {2,1}, beta = {3,2};

    // Initialize buffers 
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            a[i*lda+j].real = 2; a[i*lda+j].imag = 1;
            b[i*ldb+j].real = 0; b[i*ldb+j].imag = 0;
            c[i*ldc+j].real = 1; c[i*ldc+j].imag = 2;
            if (i == j)
            {
                b[i*ldb+j].real = j; b[i*ldb+j].imag = j;
            }
        }
	}

    // Bind buffers
    ampblas_result re = AMPBLAS_OK;
	EXECUTE_IF_OK(re, ampblas_bind(a, m * lda * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(b, k * ldb * sizeof(T)));
	EXECUTE_IF_OK(re, ampblas_bind(c, m * ldc * sizeof(T)));

    // Run test
    EXECUTE_KERNEL_IF_OK(re, ampblas_xgemm(AmpblasRowMajor, AmpblasNoTrans, AmpblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
	EXECUTE_IF_OK(re, ampblas_synchronize(c, m * ldc * sizeof(T)));

    // Unbind buffers
	ampblas_unbind(a);
	ampblas_unbind(b);
	ampblas_unbind(c);

    // Verify result
    if (re != AMPBLAS_OK)
    {
        return false;
    }
	for (int i=0; i<m; i++)
	{
        for (int j=0; j<n; j++)
        {
            T expected = {-j-1,7*j+8}; // (2,1) *(j,3*j) + (3,2)*(1,2)
            T actual  = c[i*ldc+j];

            if (!compare_equal(&actual, &expected))
            {
                return false;
            }
        }
	}

    return true;
}

