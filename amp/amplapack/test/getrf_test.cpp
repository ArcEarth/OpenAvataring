#include <vector>
#include <algorithm>
#include <iostream>

#include "amplapack_test.h"
#include "ampxlapack.h"

// host GEMM used for reconstruction
#include "lapack_host.h"

template <typename value_type>
double gflops(double sec, double m, double n)
{
    const double c = is_complex<value_type>::value ? double(4) : double(1);
    double operation_count = c * m*n*n - double(1)/double(3)*n*n*n;

    return operation_count / (sec * double(1e9));
}

template <typename value_type>
void do_getrf_test(int m, int n, int lda_offset = 0)
{
    // header
    std::cout << "Testing " << type_prefix<value_type>() << "GETRF for M=" << m << " N=" << n << " LDA=" << m+lda_offset << "... ";

    // performance timer
    high_resolution_timer timer;

    // create data
    int k = std::min(m,n);
    int lda = m + lda_offset;
    std::vector<value_type> a(lda*n, value_type(1));
    std::vector<int> ipiv(k);

    // fill with random values
    std::for_each(a.begin(), a.end(), [&](value_type& val) {
        val = random_value(value_type(-1), value_type(1));
    });

    // adjust diagonal for stability
    for (int i = 0; i < (lda*n); i += (lda+1))
        a[i] = random_value(value_type(1), value_type(2));

    // mark data outside of the leading dimension for debugging purposes
    for (int j = 0; j < n; j++)
        for (int i = 0; i < lda; i++)
            if (i >= n)
                a[j*lda+i] = value_type(-1);

    // backup a for reconstruction purposes
    std::vector<value_type> a_in(a);
    
    int info;

    timer.restart();
    amplapack_status status = amplapack_getrf(m, n, cast(a.data()), lda, ipiv.data(), &info);
    double sec = timer.elapsed();

    switch(status)
    {
    case amplapack_success:
        std::cout << "Success!";
        break;
    case amplapack_data_error:
        std::cout << "Date Error @ " << info << std::endl;
        break;
    case amplapack_argument_error:
        std::cout << "Argument Error @ " << -info << std::endl;
        break;
    case amplapack_runtime_error:
        std::cout << "Runtime Error" << std::endl;
        break;
    case amplapack_memory_error:
        std::cout << "Insuffecient Memory" << std::endl;
        break;
    default:
        std::cout << "Unexpected Error?" << std::endl;
        break;
    }

    if (status == amplapack_success)
    {
        // swap on a
        laswp(k, a_in.data(), lda, 1, k, ipiv.data(), 1);

        // extract l and u
        std::vector<value_type> l(a);
        std::vector<value_type>& u = a;
        
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                if (j > i)
                    l[j*lda+i] = value_type();

                if (j == i)
                    l[j*lda+i] = value_type(1);

                if (j < i)
                    u[j*lda+i] = value_type();
            }
        }

        // a = a - l*u
        gemm('n', 'n', m, n, k, value_type(1), l.data(), lda, u.data(), lda, value_type(-1), a_in.data(), lda);

        // norm
        std::cout << " Error = " << one_norm(n, n, a_in.data(), lda) << " GFLOPs = " << gflops<value_type>(sec, m, n) << std::endl;
    }
}

void getrf_test()
{
    // quick tests
    do_getrf_test<float>(1024, 1024); 
    do_getrf_test<fcomplex>(1024, 1024);
}