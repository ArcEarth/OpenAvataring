/*----------------------------------------------------------------------------
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
 * amax_test_util.h
 *
 * Tools used by testbench to implement testing framework.
 *
 *---------------------------------------------------------------------------*/

#pragma once

#include "ampblas_defs.h"

#include "ampblas_complex.h"
#include "ampcblas_complex.h"
#include "ampcblas_runtime.h"

#include "cblas_wrapper.h"

#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <limits>

#include <assert.h>

//
// Explicit Type Relations 
//

typedef ampblas::complex<float> complex_float;
typedef ampblas::complex<double> complex_double;

template <typename value_type> 
struct get_promoted_type 
{ 
	typedef typename value_type value; 
};

template <> struct get_promoted_type<complex_float> { typedef complex_double value; };
template <> struct get_promoted_type<float>         { typedef         double value; };

template <typename value_type> 
struct get_real_type 
{ 
	typedef typename value_type value; 
};

template <> struct get_real_type<complex_float>  { typedef float  value; };
template <> struct get_real_type<complex_double> { typedef double value; };

template <typename value_type> 
struct get_complex_type 
{ 
	typedef typename value_type value; 
};

template <> struct get_complex_type<float>  { typedef complex_float  value; };
template <> struct get_complex_type<double> { typedef complex_double value; };

//
// Testing Type Conversions
// 

// test type --> cblas types
template <typename T> struct cblas_type                 { typedef typename T            type; }; 
template <>           struct cblas_type<complex_float>  { typedef cblas::complex_float  type; };
template <>           struct cblas_type<complex_double> { typedef cblas::complex_double type; };
template <typename T> typename cblas_type<T>::type* cblas_cast(T* ptr) { return reinterpret_cast<typename cblas_type<T>::type*>(ptr); }
template <typename T> typename cblas_type<T>::type cblas_cast(T val) { return *reinterpret_cast<typename cblas_type<T>::type*>(&val); }

// ampblas (used in testing) option --> cblas option
inline cblas::transpose cblas_cast(const AMPBLAS_TRANSPOSE& trans)
{
    switch(trans)
    {
    case AmpblasNoTrans:
        return cblas::transpose::no_trans;
    case AmpblasTrans:
        return cblas::transpose::trans;
    case AmpblasConjTrans:
    default:
        return cblas::transpose::conj_trans;
    }
}

inline cblas::side cblas_cast(const AMPBLAS_SIDE& side)
{
    switch(side)
    {
    case AmpblasLeft:
        return cblas::side::left;
    case AmpblasRight:
    default:
        return cblas::side::right;
    }
}

inline cblas::uplo cblas_cast(const AMPBLAS_UPLO& uplo)
{
    switch(uplo)
    {
    case AmpblasUpper:
        return cblas::uplo::upper;
    case AmpblasLower:
    default:
        return cblas::uplo::lower;
    }
}

inline cblas::diag cblas_cast(const AMPBLAS_DIAG& diag)
{
    switch(diag)
    {
    case AmpblasNonUnit:
        return cblas::diag::non_unit;
    case AmpblasUnit:
    default:
        return cblas::diag::unit;
    }
}

// test types --> ampcblas types
template <typename T> struct ampcblas_type                 { typedef typename T       type; }; 
template <>           struct ampcblas_type<complex_float>  { typedef ampblas_fcomplex type; };
template <>           struct ampcblas_type<complex_double> { typedef ampblas_dcomplex type; };
template <typename T> typename ampcblas_type<T>::type* ampcblas_cast(T* ptr) { return reinterpret_cast<typename ampcblas_type<T>::type*>(ptr); }
template <typename T> typename ampcblas_type<T>::type ampcblas_cast(T val) { return *reinterpret_cast<typename ampcblas_type<T>::type*>(&val); }

//
// Named Type Helper
//
    
#define AMPBLAS_STRINGIZE(X) AMPBLAS_DO_STRINGIZE(X)
#define AMPBLAS_DO_STRINGIZE(X) #X
#define AMPBLAS_NAMED_TYPE(x) make_named_type( AMPBLAS_STRINGIZE(x), x )

template <typename T>
struct named_type
{
    std::string name_;
    const T& val_;

    named_type( std::string name, const T& val ) 
        : name_(name), val_(val) {}
};

template<class T1, class T2, class T3>
std::basic_ostream<T1,T2>& operator<<(std::basic_ostream<T1,T2> &os, const named_type<T3>& x) {
    os << x.name_ << ":" << x.val_ << " ";
    return os;
}

template <typename T>
named_type<T> make_named_type( std::string name, const T& val ) { return named_type<T>(name,val); } 

template <typename T>
inline char blas_prefix();

template <> inline char blas_prefix<float>() { return 'S'; }
template <> inline char blas_prefix<double>() { return 'D'; }
template <> inline char blas_prefix<complex_float>() { return 'C'; }
template <> inline char blas_prefix<complex_double>() { return 'Z'; }

//
// Templated Loops
//

template <typename G, class T1, class T2>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    g.push_back( G(t1,t2) );
}

template <typename G, class T1, class T2, class T3>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    g.push_back( G(t1,t2,t3) );
}

template <typename G, class T1, class T2, class T3, class T4>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    g.push_back( G(t1,t2,t3,t4) );
}

template <typename G, class T1, class T2, class T3, class T4, class T5>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s, const T5& t5s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    for ( auto& t5 : t5s )
    g.push_back( G(t1,t2,t3,t4,t5) );
}

template <typename G, class T1, class T2, class T3, class T4, class T5, class T6>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s, const T5& t5s, const T6& t6s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    for ( auto& t5 : t5s )
    for ( auto& t6 : t6s )
    g.push_back( G(t1,t2,t3,t4,t5,t6) );
}

template <typename G, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s, const T5& t5s, const T6& t6s, const T7& t7s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    for ( auto& t5 : t5s )
    for ( auto& t6 : t6s )
	for ( auto& t7 : t7s )
    g.push_back( G(t1,t2,t3,t4,t5,t6,t7) );
}

template <typename G, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s, const T5& t5s, const T6& t6s, const T7& t7s, const T8& t8s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    for ( auto& t5 : t5s )
    for ( auto& t6 : t6s )
	for ( auto& t7 : t7s )
	for ( auto& t8 : t8s )
    g.push_back( G(t1,t2,t3,t4,t5,t6,t7,t8) );
}

template <typename G, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s, const T5& t5s, const T6& t6s, const T7& t7s, const T8& t8s, const T9& t9s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    for ( auto& t5 : t5s )
    for ( auto& t6 : t6s )
	for ( auto& t7 : t7s )
	for ( auto& t8 : t8s )
	for ( auto& t9 : t9s )
    g.push_back( G(t1,t2,t3,t4,t5,t6,t7,t8,t9) );
}

template <typename G, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
void container_exploder(std::vector<G>& g, const T1& t1s, const T2& t2s, const T3& t3s, const T4& t4s, const T5& t5s, const T6& t6s, const T7& t7s, const T8& t8s, const T9& t9s, const T10& t10s)
{
    for ( auto& t1 : t1s )
    for ( auto& t2 : t2s )
    for ( auto& t3 : t3s )
    for ( auto& t4 : t4s )
    for ( auto& t5 : t5s )
    for ( auto& t6 : t6s )
	for ( auto& t7 : t7s )
	for ( auto& t8 : t8s )
	for ( auto& t9 : t9s )
	for ( auto& t10 : t10s )
    g.push_back( G(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10) );
}

//
// Exception Handeling
//

class ampblas_test_runtime_exeception
{
public:
    ampblas_test_runtime_exeception(const ampblas_result& e) 
        : e_(e) 
    {}

    ampblas_result get() const 
    { 
        return e_; 
    }

private:
    ampblas_result e_;
};

class ampblas_error_checker
{
public:
    ampblas_error_checker() {}

    ampblas_error_checker(const ampblas_result& result) { 
        check_error(result); 
    }

    void check_last()
    {
        ampblas_result last_error = ampblas_get_last_error();
        check_error(last_error);
    }

private:
    void check_error(const ampblas_result& result) 
    {
        if (result != AMPBLAS_OK)
            throw ampblas_test_runtime_exeception(result);
    }
};

//
// Simple Testing Vector
// 

template <typename value_type>
class test_vector
{
public:
    typedef typename std::vector<value_type> container_type;

    test_vector(int n, int inc = 1)
        : n_(n), inc_(abs(inc)), data_(n_*inc_, value_type())
    {}

    ~test_vector(){}

    // size in bytes
    size_t size() const
    {
        return n_*inc_*sizeof(value_type);
    }

    // vector size
    int n() const 
    {
        return n_;
    }

    int inc() const 
    {
        return inc_;
    }

    value_type* data()
    { 
        return &data_.front();
    }

    const value_type* data() const 
    {
        return &data_.front();
    }

    const value_type& operator[](int i) const 
    {
        return data_[i*inc_];
    }

    value_type& operator[](int i) 
    {
        return data_[i*inc_];
    }

    typename container_type::iterator begin() 
    {
        return data_.begin();
    }

    typename container_type::iterator end() 
    {
        return data_.end();
    }

    container_type& container()
    { 
        return data_; 
    } 

private:
    int n_;
    int inc_;
    container_type data_;
};

template <typename value_type>
class ampblas_test_vector : public test_vector<value_type>
{
public:
    ampblas_test_vector(int n, int inc = 1)
        : test_vector<value_type>(n, inc), is_bound(false)
    {
        err = ampblas_bind(test_vector<value_type>::data(), test_vector<value_type>::size());
        is_bound = true;
    }

    ampblas_test_vector(const test_vector<value_type>& rhs)
        : test_vector<value_type>(rhs), is_bound(false)
    {
        err = ampblas_bind(test_vector<value_type>::data(), test_vector<value_type>::size());
        is_bound = true;
    }

    void unbind()
    {
        if (is_bound)
            ampblas_unbind(test_vector<value_type>::data());
    }

    void synchronize()
    {
        if (is_bound)
            ampblas_synchronize(test_vector<value_type>::data(), test_vector<value_type>::size());
    }

    ~ampblas_test_vector()
    {
        if (is_bound)
            unbind();            
    }

private:
    bool is_bound;
    ampblas_error_checker err;
};

//
// Simple Testing Matrix
//

template <typename value_type>
class test_matrix
{
public:
    typedef typename std::vector<value_type> container_type;

    test_matrix(int m, int n, int ld = 0)
        : m_(m), n_(n), ld_( ld ? ld : m ), data_(ld_*n_, value_type())
    {}

    ~test_matrix(){}

    // size in bytes
    size_t size() const 
    {
        return ld_ * n_ * sizeof(value_type);
    }

    int m() const 
    {
        return m_;
    }
    
    int n() const 
    {
        return n_;
    }

    int ld() const 
    {
        return ld_;
    }

    value_type* data() 
    { 
        return &data_.front();
    }

    const value_type* data() const 
    {
        return &data_.front();
    }

    const value_type& operator()(int i, int j) const
    {
        return data_[i + j*ld_];
    }

    value_type& operator()(int i, int j)
    {
        return data_[i + j*ld_];
    }

    typename container_type::iterator begin() 
    {
        return data_.begin();
    }

    typename container_type::iterator end() 
    {
        return data_.end();
    }

    container_type& container()
    { 
        return data_; 
    } 

private:
    int m_, n_, ld_;
    container_type data_;
};

template <typename value_type>
class ampblas_test_matrix : public test_matrix<value_type>
{
public:

    ampblas_test_matrix(int m, int n, int ld = 0)
        : test_matrix<value_type>(m, n, ld), is_bound(false)
    {
        err = ampblas_bind(test_matrix<value_type>::data(), test_matrix<value_type>::size());
        is_bound = true;
    }

    ampblas_test_matrix(const test_matrix<value_type>& rhs)
        : test_matrix<value_type>(rhs), is_bound(false)
    {
        err = ampblas_bind(test_matrix<value_type>::data(), test_matrix<value_type>::size());
        is_bound = true;
    }

    void synchronize()
    {
        if (is_bound)
            ampblas_synchronize(test_matrix<value_type>::data(), test_matrix<value_type>::size());
    }

    void unbind()
    {
        ampblas_unbind(test_matrix<value_type>::data());
        is_bound = false;
    }

    ~ampblas_test_matrix()
    {
        if (is_bound)
            unbind();
    }

private:
    bool is_bound;
    ampblas_error_checker err;
};

template <typename value_type>
value_type random_value(value_type min, value_type max)
{
    value_type val = value_type(rand()) / value_type(RAND_MAX);
    val *= (max-min);
    val += min;

    return val;
}

template <typename value_type>
ampblas::complex<value_type> random_value(ampblas::complex<value_type> min, ampblas::complex<value_type> max)
{
    // only real values used to calculate random range
    return ampblas::complex<value_type>(random_value(min.real(),max.real()), random_value(min.real(),max.real()));
}

template <typename value_type>
void randomize_vector(std::vector<value_type>& vec, value_type min = value_type(-1), value_type max = value_type(1))
{    
    std::for_each( vec.begin(), vec.end(), [&](value_type& val) {
        val = random_value(min,max);
    });
}

template <typename value_type>
void randomize(test_matrix<value_type>& mat, value_type min = value_type(-1), value_type max = value_type(1))
{
    randomize_vector(mat.container(), min, max);
};

template <typename value_type>
void randomize(test_vector<value_type>& vec, value_type min = value_type(-1), value_type max = value_type(1))
{
    randomize_vector(vec.container(), min, max);
}

template <typename value_type>
void ones(test_matrix<value_type>& mat)
{
    std::fill(mat.container().begin(), mat.container().end(), value_type(1));
};

template <typename value_type>
void ones(test_vector<value_type>& vec)
{
    std::fill(vec.container().begin(), vec.container().end(), value_type(1));
}
