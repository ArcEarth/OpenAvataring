#ifndef AMPBLAS_TUNE_H
#define AMPBLAS_TUNE_H

#include <amp.h>
#include <string>

#include "../../ampblas/inc/ampblas_complex.h"
#include "../../ampblas/inc/ampblas_defs.h"

// these are defined in amp.h somewhere...
#ifdef max
#undef max
#endif 

#ifdef min
#undef min
#endif 

// declare namespace
#define TUNE_NAMESPACE_BEGIN namespace tune {
#define TUNE_NAMESPACE_END } // namespace tune

TUNE_NAMESPACE_BEGIN

// 
using ampblas::transpose;

// complex support
typedef ampblas::complex<float> fcomplex;
typedef ampblas::complex<double> dcomplex;

template <typename value_type> inline std::string type_name() { return typeid(value_type).name(); }
template <> inline std::string type_name<fcomplex>() { return "fcomplex"; }
template <> inline std::string type_name<dcomplex>() { return "dcomplex"; }

// GFLOPs helpers
template <typename value_type> struct gemm_flops_multiplier{ static const int value = 2;};
template <typename value_type> struct gemm_flops_multiplier<ampblas::complex<value_type>> { static const int value = 8;};

template <enum class transpose t> struct trans_name;
template<> struct trans_name<transpose::no_trans> { static std::string value() { return "transpose::no_trans"; } };
template<> struct trans_name<transpose::trans> { static std::string value() { return "transpose::trans"; } };
template<> struct trans_name<transpose::conj_trans> { static std::string value() { return "transpose::conj_trans"; } };

// return the BLAS transpose character option  
template <enum class transpose t> struct trans_prefix;
template <> struct trans_prefix<transpose::no_trans> { static const char value = 'n'; }; 
template <> struct trans_prefix<transpose::trans> { static const char value = 't'; };
template <> struct trans_prefix<transpose::conj_trans> { static const char value = 'c'; };

// return the BLAS type prefix
template <typename value_type> struct type_prefix;
template <> struct type_prefix<float> { static const char value = 's'; };
template <> struct type_prefix<double> { static const char value = 'd'; };
template <> struct type_prefix<fcomplex> { static const char value = 'c'; };
template <> struct type_prefix<dcomplex> { static const char value = 'z'; };

// number of iterations to test
static const int batch_size = 4;

// internal exception
struct tune_failure_exception
{
    int id;
    tune_failure_exception(int id)
        : id(id) {}
};

// option set
struct static_options
{
    int m_block;
    int n_block;
    int k_block;
    int c_m; 
    int c_n; 
    int a_m; 
    int a_n; 
    int b_m; 
    int b_n;
    int padding;

    // default ctor
    static_options() {}

    // main ctor
    static_options(int m_block, int n_block, int k_block, int c_m, int c_n, int a_m, int a_n, int b_m, int b_n, int padding)
        : m_block(m_block), n_block(n_block), k_block(k_block), c_m(c_m), c_n(c_n), a_m(a_m), a_n(a_n), b_m(b_m), b_n(b_n), padding(padding)
    {}
};

// operator <<
template<typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& bos, const static_options& opt) 
{
    bos << opt.m_block << ",";
    bos << opt.n_block << ",";
    bos << opt.k_block << ",";
    bos << opt.c_m << ",";
    bos << opt.c_n << ",";
    bos << opt.a_m << ",";
    bos << opt.a_n << ",";
    bos << opt.b_m << ",";
    bos << opt.b_n << ",";
    bos << opt.padding;

    return bos;
}

// results
struct search_result 
{
    char type;
    char transa;
    char transb;
    static_options options; 
    double gflops;

    // minimal ctor
    search_result(char type, char transa, char transb)
        : type(type), transa(transa), transb(transb)
    {}

    // default ctor
    search_result(char type, char transa, char transb, static_options options, double gflops)
        : type(type), transa(transa), transb(transb), options(options), gflops(gflops)
    {}

    // operator >
    bool operator<(const search_result& rhs) 
    {
        return gflops > rhs.gflops;
    }
};

// global result vector
std::vector<search_result>& current_results();

TUNE_NAMESPACE_END

#endif // AMPBLAS_TUNE_H
