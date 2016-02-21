#include <algorithm>
#include <iostream>

#include "tune.h"
#include "template.h"
#include "host_gemm.h"

TUNE_NAMESPACE_BEGIN

std::vector<search_result>& current_results()
{
    static std::vector<search_result> current_results;
    return current_results;
}

template <typename value_type>
value_type random_value()
{
    value_type min = value_type(-1); 
    value_type max = value_type(1);

    value_type val = value_type(rand()) / value_type(RAND_MAX);
    val *= (max-min);
    val += min;
    return val;
}

template <> fcomplex random_value<fcomplex>() { return fcomplex( random_value<float>(), random_value<float>() ); }
template <> dcomplex random_value<dcomplex>() { return dcomplex( random_value<float>(), random_value<float>() ); }

void print_header(int n) 
{
    std::cout << "type" << ","
        << "transa" << "," 
        << "transb" << ","
        << "m_block" << "," 
        << "n_block" << "," 
        << "k_block" << ","
        << "m_c_tile" << "," 
        << "n_c_tile" << ","
        << "m_a_tile" << "," 
        << "n_a_tile" << ","
        << "m_b_tile" << "," 
        << "n_b_tile" << ","
        << "a_padding" << ","
        << "b_padding" << ","
        << "thread_count" << ","
        << "registers" << ","
        << "shared_memory" << " KB,"
        << "gflops_" << n << std::endl;
}

template <typename value_type, enum class transpose transa, enum class transpose transb>
search_result build_test(int m, int n, int k, int offset = 0)
{
    // scalars
    value_type alpha = value_type(1);
    value_type beta = value_type(0);

    int row_a = m;
    int col_a = k;

    int row_b = k;
    int col_b = n;

    if (transa != transpose::no_trans)
        std::swap(row_a, col_a);

    if (transb != transpose::no_trans)
        std::swap(row_b, col_b);

    try 
    {
        // get a fresh accelerator view to reset some resources
        concurrency::accelerator_view av = concurrency::accelerator().create_view();

        // reset the global results if starting a new test 
        if (offset == 0)
            current_results().clear();

        // raw data
        std::vector<value_type> a(m*k, value_type(1));
        std::vector<value_type> b(k*n, value_type(1));
        std::vector<value_type> c(m*n, value_type(1));

        // fill with data
        std::for_each(a.begin(), a.end(), [](value_type& x){ x = random_value<value_type>(); });
        std::for_each(b.begin(), b.end(), [](value_type& x){ x = random_value<value_type>(); });
        std::for_each(c.begin(), c.end(), [](value_type& x){ x = random_value<value_type>(); });

        // amp data
        concurrency::array<value_type,2> a_array(row_a, col_a, a.data(), av);
        concurrency::array<value_type,2> b_array(row_b, col_b, b.data(), av);
        concurrency::array<value_type,2> c_array(m, n, c.data(), av);

        // reference solution (row major)
        std::vector<value_type> c_ref = c;
        host::matrix_matrix_multiply(trans_prefix<transa>::value, trans_prefix<transb>::value, m, n, k, alpha, a.data(), col_a, b.data(), col_b, beta, c_ref.data(), n);

        // amp views
        concurrency::array_view<const value_type,2> a_view(a_array);
        concurrency::array_view<const value_type,2> b_view(b_array);
        concurrency::array_view<value_type,2> c_view(c_array);

        // run search space sweeps
        do_test<value_type, transa, transb>(av, alpha, a_view, b_view, beta, c_view, c_ref, offset);
    }
    catch(const tune_failure_exception& e)
    {
        // restart testing; with the new offset
        build_test<value_type, transa, transb>(m, n, k, e.id + 1);
    }
    catch(const concurrency::runtime_exception& e)
    {
        std::cout << "C++ AMP Runtime Exception: " << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "Unknown Error!" << std::endl;
    }

    // return the best
    if (!current_results().empty())
    {
        std::sort(current_results().begin(), current_results().end());
        return current_results().front();
    }
    else
    {
        return search_result(type_prefix<value_type>::value, trans_prefix<transa>::value, trans_prefix<transb>::value);
    }
}

TUNE_NAMESPACE_END

int main()
{
    using namespace tune;

    // TODO: support a range of data sizes or pass in as an argument
    const int m = 2048;
    const int n = 2048;
    const int k = 2048;

    // formating
    print_header(m);

    // result vector
    std::vector<search_result> top_finds;

    // testing
    top_finds.push_back( build_test<float, transpose::no_trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<float, transpose::no_trans, transpose::trans>(m,n,k) );
    top_finds.push_back( build_test<float, transpose::trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<float, transpose::trans, transpose::trans>(m,n,k) );
     
    top_finds.push_back( build_test<double, transpose::no_trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<double, transpose::no_trans, transpose::trans>(m,n,k) );
    top_finds.push_back( build_test<double, transpose::trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<double, transpose::trans, transpose::trans>(m,n,k) );
    
    top_finds.push_back( build_test<fcomplex, transpose::no_trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<fcomplex, transpose::no_trans, transpose::trans>(m,n,k) );
    top_finds.push_back( build_test<fcomplex, transpose::trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<fcomplex, transpose::trans, transpose::trans>(m,n,k) );
    
    top_finds.push_back( build_test<dcomplex, transpose::no_trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<dcomplex, transpose::no_trans, transpose::trans>(m,n,k) );
    top_finds.push_back( build_test<dcomplex, transpose::trans, transpose::no_trans>(m,n,k) );
    top_finds.push_back( build_test<dcomplex, transpose::trans, transpose::trans>(m,n,k) );

    std::cout <<  "--- TOP RESULTS ---" << std::endl;
    std::for_each(top_finds.begin(), top_finds.end(), [](search_result& r)
    {
        std::cout << r.type << "gemm_" << r.transa << r.transb << " = {" << r.options << "} @ " << r.gflops << std::endl;
    });
   
    return 0;
}
