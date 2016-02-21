#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <future>

#include "tune.h"

TUNE_NAMESPACE_BEGIN

template <typename value_type, enum class transpose transa, enum class transpose transb>
std::string header_name()
{
    std::stringstream ss;
    ss << type_prefix<value_type>::value << "gemm_" << trans_prefix<transa>::value << trans_prefix<transb>::value; 
    return ss.str();
}

struct dx11
{
    // DX11 shared memory maximum
    static const int max_shared_memory() { return 32768; }
   
    template <typename value_type>
    struct max_registers;
};

// D3D11_COMMONSHADER_TEMP_REGISTER_COUNT = 4096
// Common-shader core (four 32-bit components) temp-register count (r# + indexable x#[n])
template <> struct dx11::max_registers<float> { static const int value = 16384; };
template <> struct dx11::max_registers<double> { static const int value = 8192; };
template <> struct dx11::max_registers<fcomplex> { static const int value = 8192; };
template <> struct dx11::max_registers<dcomplex> { static const int value = 4096; };

template <typename value_type, enum class transpose transa, enum class transpose transb>
void generate_search_space()
{
    // common strings
    const std::string parameter_string = "av, alpha, a, b, beta, c, c_ref, offset";
    const std::string data_extension = ".data";
    const std::string data_folder = "data";
    const std::string header_warning = "this file was automatically generated; edit at your own risk";

    //
    // Heuristics to limit search space
    //

    // there are potentially thousands of valid options; the compiler will "overflow" if this number is too large (around 300)
    const size_t max_search_space = 256;

    // work block stepping
    const int max_work_block_size = 256;
    const int work_block_step = 2;
    const int work_block_min = 4;

    // thread stepping
    const int min_thread_tile_size = 128;
    const int max_thread_tile_size = 768;
    const int thread_step = 32;
    
    // tile stepping
    const int tile_step = 2;

    // shared memory padding
    const bool use_padding = true;

    // exclude implementations with poor resource usage
    const int min_registers = 3;

    // estimated number of registers needed for control
    const int control_registers = 10;

    //
    // Generate header name
    //

    std::string file_name = header_name<value_type, transa, transb>();
    std::ofstream file( data_folder + "/" + file_name + data_extension);

    //
    // Loop over all valid option sets
    // 

    std::vector<static_options> valid_options;

    // work block 
    for (int m_block = work_block_step; m_block <= max_work_block_size; m_block += work_block_step)
    {
        for (int n_block = work_block_step; n_block <= max_work_block_size; n_block += work_block_step)
        {
            for (int k_block = work_block_step; k_block <= max_work_block_size; k_block += work_block_step)
            {
                // calculate shared memory usage
                const int a_padding = (use_padding && transa != transpose::no_trans ? 1 : 0);
                const int b_padding = (use_padding && transb != transpose::no_trans ? 1 : 0);
                int shared_mem = sizeof(value_type) * ((m_block*(k_block + a_padding)) + (k_block*(n_block + b_padding)));

                // skip now if requesting too much shared memory
                if (shared_mem > dx11::max_shared_memory())
                    continue;

                // thread counts
                for (int n_threads = min_thread_tile_size; n_threads <= max_thread_tile_size; n_threads += thread_step)
                {
                    // tile configurations
                    for (int c_m = work_block_min; c_m <= n_threads; c_m += tile_step)
                    {
                        int c_n = n_threads / c_m;

                        if (c_n < work_block_min)
                            continue;

                        if (c_m*c_n != n_threads)
                            continue;

                        if (m_block % c_m != 0 || n_block % c_n != 0)
                            continue;

                        for (int a_m = work_block_min; a_m <= n_threads; a_m += tile_step)
                        {
                            int a_n = n_threads / a_m;

                            if (a_n < work_block_min)
                                continue;

                            if (a_m*a_n != n_threads)
                                continue;

                            if (m_block % (transa == transpose::no_trans ? a_m : a_n) != 0 || k_block % (transa == transpose::no_trans ? a_n : a_m) != 0)
                                continue;

                            for (int b_m = work_block_min; b_m <= n_threads; b_m += tile_step)
                            {
                                int b_n = n_threads / b_m;

                                if (b_n < work_block_min)
                                    continue;

                                if (b_m*b_n != n_threads)
                                    continue;

                                if (k_block % (transb == transpose::no_trans ? b_m : b_n) != 0 || n_block % (transb == transpose::no_trans ? b_n : b_m) != 0)
                                    continue;

                                // calculate register usage
                                const int m_thread_work = m_block / c_m;
                                const int n_thread_work = n_block / c_n;
                                const int registers = (m_thread_work*n_thread_work) + m_thread_work + n_thread_work;         

                                // too many registers seems to cause kernel crashes / hangs
                                if ( n_threads*(registers+control_registers) > dx11::max_registers<value_type>::value)
                                    continue;

                                // too few registers may cause poor performance on some systems
                                if (registers < min_registers)
                                    continue;

                                // add options if they meet all of the requirements
                                valid_options.push_back(static_options(m_block, n_block, k_block, c_m, c_n, a_m, a_n, b_m, b_n, use_padding));
                            }
                        }
                    }
                }
            }
        }
    }

    // shuffle 
    std::random_shuffle(valid_options.begin(), valid_options.end());

    //
    // Write a sub set of the options to the headers
    // 
        
    const size_t count = std::min(max_search_space, valid_options.size());

    std::cout << "Writing " << count << " of a possible " << valid_options.size() << " parameter sets to 'data/" << file_name << ".data'" << std::endl;

    // write to data file
    file << "/* " << header_warning << " */" << std::endl << std::endl;
    for (size_t i = 0; i < count; i++)
    {
        const static_options& opt = valid_options[i];
        file << "benchmark_gemm<" 
            << i << ","                                                // test_id
            << type_name<value_type>() << ","                          // type
            << "true,"                                                 // guard option
            << trans_name<transa>::value() << ","                      // transa
            << trans_name<transb>::value() << ","                      // transb
            << opt << ">(" << parameter_string <<  ");" << std::endl;  // tuning and params
    }

}

TUNE_NAMESPACE_END

int main()
{
    using namespace tune;

    std::vector<std::future<void>> futures;

    futures.push_back(std::async(generate_search_space<float, transpose::no_trans, transpose::no_trans>));
    futures.push_back(std::async(generate_search_space<float, transpose::no_trans, transpose::trans>)); 
    futures.push_back(std::async(generate_search_space<float, transpose::trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<float, transpose::trans, transpose::trans>));

    futures.push_back(std::async(generate_search_space<double, transpose::no_trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<double, transpose::no_trans, transpose::trans>));
    futures.push_back(std::async(generate_search_space<double, transpose::trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<double, transpose::trans, transpose::trans>));

    futures.push_back(std::async(generate_search_space<fcomplex, transpose::no_trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<fcomplex, transpose::no_trans, transpose::trans>));
    futures.push_back(std::async(generate_search_space<fcomplex, transpose::trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<fcomplex, transpose::trans, transpose::trans>));

    futures.push_back(std::async(generate_search_space<dcomplex, transpose::no_trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<dcomplex, transpose::no_trans, transpose::trans>));
    futures.push_back(std::async(generate_search_space<dcomplex, transpose::trans, transpose::no_trans>)); 
    futures.push_back(std::async(generate_search_space<dcomplex, transpose::trans, transpose::trans>));

    // wait for all
    std::for_each( futures.begin(), futures.end(), [](std::future<void>& f){ f.wait(); } );
}
