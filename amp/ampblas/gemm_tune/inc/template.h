#include <iostream>
#include <vector>

#include "tune.h"
#include "high_resolution_timer.h"

// use the GEMM implementation from actual AMPBLAS project
#include "detail/gemm.h"

TUNE_NAMESPACE_BEGIN

// templated test entry point
template <typename value_type, enum class transpose transa, enum class transpose transb>
void do_test(concurrency::accelerator_view& av, value_type alpha, const concurrency::array_view<const value_type,2>& a, const concurrency::array_view<const value_type,2>& b, value_type beta, const concurrency::array_view<value_type,2>& c, const std::vector<value_type>& c_ref, int offset);

// templated test main
template <int test_id, typename value_type, bool guarded, enum class transpose transa, enum class transpose transb, int m_block, int n_block, int k_block, int m_c_tile, int n_c_tile, int m_a_tile, int n_a_tile, int m_b_tile, int n_b_tile, int use_padding, typename a_type, typename b_type, typename c_type>
void benchmark_gemm(concurrency::accelerator_view& av, value_type alpha, const a_type& a, const b_type& b, value_type beta, const c_type& c, const std::vector<value_type>& c_ref, int offset)
{
    // skip to offset
    if (test_id < offset)
        return;

    // const copy used for testing
    const concurrency::accelerator_view& const_av = const_cast<const concurrency::accelerator_view&>(av);

    // sizes
    const int m = c.extent[0];  
    const int n = c.extent[1];
    const int k = (transa != transpose::no_trans ? a.extent[0] : a.extent[1]); 
       
    // static calculations
    const int thread_count = m_c_tile * n_c_tile;
    const int a_padding = (use_padding && transa != transpose::no_trans ? 1 : 0);
    const int b_padding = (use_padding && transb != transpose::no_trans ? 1 : 0);
    const int shared_memory = ((m_block * (k_block + a_padding)) + (k_block * (n_block + b_padding))) * sizeof(value_type);
    const int m_thread = m_block / m_c_tile;
    const int n_thread = n_block / n_c_tile;

    // static checks for thread usage
    static_assert(m_a_tile * n_a_tile == thread_count, "static tuning error: a tile shape must fit within register allocation");
    static_assert(m_b_tile * n_b_tile == thread_count, "static tuning error: b tile shape must fit within register allocation");
    static_assert(m_c_tile * n_c_tile == thread_count, "static tuning error: c tile shape must fit within register allocation");
    
    // static checks for block usage
    static_assert( m_block % (transa == transpose::no_trans ? m_a_tile : n_a_tile) == 0, "static tuning error: a tile must evenly divide into [m x k] work block");
    static_assert( k_block % (transa == transpose::no_trans ? n_a_tile : m_a_tile) == 0, "static tuning error: a tile must evenly divide into [m x k] work block");
    static_assert( k_block % (transb == transpose::no_trans ? m_b_tile : n_b_tile) == 0, "static tuning error: b tile must evenly divide into [k x n] work block");
    static_assert( n_block % (transb == transpose::no_trans ? n_b_tile : m_b_tile) == 0, "static tuning error: b tile must evenly divide into [k x n] work block");
    static_assert( m_block % m_c_tile == 0, "static tuning error: c tile must evenly divide into [m x n] work block");
    static_assert( n_block % n_c_tile == 0, "static tuning error: c tile must evenly divide into [m x n] work block");

    // timer
    tune::high_resolution_timer timer;
    double t_sec = 0;

    // print out parameter listing
    std::cout << type_name<value_type>() <<  ","
        << trans_prefix<transa>::value << "," 
        << trans_prefix<transb>::value <<  ","
        << m_block << "," 
        << n_block << "," 
        << k_block << ","
        << m_c_tile << "," 
        << n_c_tile << ","
        << m_a_tile << "," 
        << n_a_tile << ","
        << m_b_tile << "," 
        << n_b_tile << ","
        << a_padding << ","
        << b_padding << ","
        << thread_count << ","
        << (m_thread*n_thread) + m_thread + n_thread << ","
        << shared_memory/1024 << " KB,";

    // parameter local try/catch
    try
    {
        // flush the accelerator to be safe
        av.flush();

        // prime the kernel
        ampblas::_detail::gemm_kernel<guarded, transa, transb, m_block, n_block, k_block, m_c_tile, n_c_tile, m_a_tile, n_a_tile, m_b_tile, n_b_tile, use_padding>(const_av, alpha, a, b, beta, c);
        av.wait();

        // run a batch to collect timing data
        for (int i = 0; i < batch_size; i++ )
        {
            // make sure the accelerator is completely ready
            av.wait();

            // run and wait for timing results
            timer.restart();
            ampblas::_detail::gemm_kernel<guarded, transa, transb, m_block, n_block, k_block, m_c_tile, n_c_tile, m_a_tile, n_a_tile, m_b_tile, n_b_tile, use_padding>(const_av, alpha, a, b, beta, c);
            av.wait();
            t_sec += timer.elapsed();
        }

        // check answer
        std::vector<value_type> c_host(m*n);
        concurrency::copy(c, c_host.begin());

        typedef typename ampblas::real_type<value_type>::type real_type;
        real_type norm = 0;
		real_type threshold = std::numeric_limits<real_type>::epsilon()*m*k;
		
		for (int j=0; j<n; j++)
		{
			real_type sum = 0;
			for (int i=0; i<m; i++)
				sum += abs( c_ref[i+m*j] - c_host[i+m*j] );
			norm = std::max(norm, sum);
		}

        if (norm > threshold)
        {
            std::cout << "Accuracy Failure for test" << test_id << ": " << norm << " > " << threshold << std::endl;
            return;
        }
            
        // success; print out GFLOPs
        double gflops = double(batch_size)*double(gemm_flops_multiplier<value_type>::value)*double(m)*double(n)*double(k) / (t_sec*double(1e9)) ;
        std::cout << gflops << std::endl;

        // add to results array
        search_result results = search_result( type_prefix<value_type>::value, trans_prefix<transa>::value, trans_prefix<transb>::value, static_options(m_block, n_block, k_block, m_c_tile, n_c_tile, m_a_tile, n_a_tile, m_b_tile, n_b_tile, use_padding), gflops );
        current_results().push_back(results);

    }
    catch(const concurrency::accelerator_view_removed& e)
    {
        std::cout << "TDR Failure for test" << test_id << ": " << e.what() << std::endl;
        throw tune_failure_exception(test_id);
    }
    catch(...)
    {
        std::cout << "Unknown Error?" << std::endl;
    }
}

TUNE_NAMESPACE_END
