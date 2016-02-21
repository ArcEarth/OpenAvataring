#include <vector>
#include <algorithm>
#include <iostream>

#include "amplapack_test.h"
#include "ampxlapack.h"

// host GEMM used for reconstruction
#include "lapack_host.h"

template <typename value_type>
inline value_type conjugate(const value_type& value)
{
    return value;
}

template <typename value_type>
inline ampblas::complex<value_type> conjugate(const ampblas::complex<value_type>& value)
{
    return ampblas::complex<value_type>(value.real(), -value.imag());
}

template <typename value_type>
double gflops(double sec, double n)
{
    const double c = is_complex<value_type>::value ? double(4) : double(1);
    double operation_count = c * double(1)/double(3)*n*n*n;

    return operation_count / (sec * double(1e9));
}

template <typename value_type>
void do_potrf_test(char uplo, int n, int lda_offset = 0)
{
    // header
    std::cout << "Testing " << type_prefix<value_type>() << "POTRF for UPLO=" << uplo << " N=" << n << " LDA=" << n+lda_offset << "... ";

    // performance timer
    high_resolution_timer timer;

    // create data
    int lda = n + lda_offset;
    std::vector<value_type> a(lda*n, value_type(1));

    // fill with random values
    std::for_each(a.begin(), a.end(), [&](value_type& val) {
        val = random_value(value_type(0), value_type(1));
    });

    // scale diagonal
    for (int i = 0; i < (lda*n); i += (lda+1))
        a[i] = value_type(ampblas::real_type<value_type>::type(n));

    // obliterate unused half
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (i < j && uplo == 'L')
                a[j*lda+i] = value_type();
            else if (i > j && uplo == 'U')
                a[j*lda+i] = value_type();

    // mark data outside of the leading dimension for debugging purposes
    for (int j = 0; j < n; j++)
        for (int i = 0; i < lda; i++)
            if (i >= n)
                a[j*lda+i] = value_type(-1);

    // backup a for reconstruction purposes
    std::vector<value_type> a_in(a);
    
    int info;

    timer.restart();
    amplapack_status status = amplapack_potrf(uplo, n, cast(a.data()), lda, &info);
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
        // reconstruction
        if (uplo == 'L' || uplo == 'l')
        {
            // reflect Ain
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    if (j > i)
                        a_in[j*lda+i] = conjugate(a_in[i*lda+j]);


            // a = a - l*l'
            gemm('n', 'c', n, n, n, value_type(1), a.data(), lda, a.data(), lda, value_type(-1), a_in.data(), lda);
        }
        else
        {   
            // reflect Ain
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    if (i > j)
                        a_in[j*lda+i] = conjugate(a_in[i*lda+j]);

            // a = a - u'*u
            gemm('c', 'n', n, n, n, value_type(1), a.data(), lda, a.data(), lda, value_type(-1), a_in.data(), lda);
        }

        // norm
        std::cout << " Error = " << one_norm(n, n, a_in.data(), lda) << " GLFOPs = " << gflops<value_type>(sec, n) << std::endl;
    }
}

void potrf_test()
{
    // performance tests
    do_potrf_test<float>('L', 1024);
    do_potrf_test<float>('U', 1024);

    do_potrf_test<fcomplex>('L', 1024);
    do_potrf_test<fcomplex>('U', 1024);
}