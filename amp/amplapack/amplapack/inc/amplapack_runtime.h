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
 * amplapack_config.h
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPLAPACK_RUNTIME_H
#define AMPLAPACK_RUNTIME_H

#include <functional>
#include <amp.h>

#include "ampclapack.h"
#include "ampblas_complex.h"

namespace amplapack {

// options
enum class ordering {row_major, column_major};
enum class transpose {no_trans, trans, conj_trans};
enum class uplo {upper, lower};
enum class diag {non_unit, unit};
enum class side {left, right};
enum class direction {forward, backward};
enum class storage {column, row};

// exeptions
class argument_error_exception
{
public:
    argument_error_exception(unsigned int index)
        : index(index) 
    {}

    // return index of invalid parameter
    unsigned int get() const
    {
        return index;
    }

private:
    unsigned int index;
};

inline void argument_error(unsigned int index) { throw argument_error_exception(index); }

class data_error_exception
{
public:
    data_error_exception(unsigned int index)
        : index(index) 
    {}

    // return location of data error
    unsigned int get() const
    {
        return index;
    }

private:
    unsigned int index;
};

inline void data_error(unsigned int index) { throw data_error_exception(index); }

class runtime_error_exception {};

inline void runtime_error() { throw runtime_error_exception(); }

// casts to internal complex types
inline ampblas::complex<float>* amplapack_cast(amplapack_fcomplex* ptr) 
{ 
    return reinterpret_cast<ampblas::complex<float>*>(ptr); 
}

inline ampblas::complex<double>* amplapack_cast(amplapack_dcomplex* ptr)
{ 
    return reinterpret_cast<ampblas::complex<double>*>(ptr); 
}

// exception safe execution wrapper
amplapack_status safe_call_interface(std::function<void(concurrency::accelerator_view& av)>& functor, int& info);

// creates a row or column vector from a 2d array with either the 1st or 2nd dimension being 1 
template <typename value_type>
class subvector_view
{
public:

    // match some of the interface of array_view
    typedef typename value_type value_type;
    concurrency::extent<1> extent;

    subvector_view(const concurrency::array_view<value_type,2>& base_view) restrict(cpu,amp)
    : base_view(base_view)
	{
        //
        if (base_view.extent[1] == 1)
        {
            direction = 0;
        }
        else if (base_view.extent[0] == 1)
        {
            direction = 1;            
        }

        extent = concurrency::extent<1>(base_view.extent[direction]);
	}

	~subvector_view() restrict(cpu,amp) {}

	value_type& operator[] (const concurrency::index<1>& idx) const restrict(cpu,amp)
	{
        concurrency::index<2> idx_2d = (direction == 0 ? concurrency::index<2>(idx[0],0) : concurrency::index<2>(0,idx[0]));
		return base_view[idx_2d];
	}

private:

    // the dimension of the data:
    //   0 is a row vector
    //   1 is a column vector
    unsigned int direction;
    concurrency::array_view<value_type,2> base_view;
};

template <typename value_type>
subvector_view<value_type> make_subvector_view(const concurrency::array_view<value_type,2>& base_view) restrict(cpu, amp)
{
    return subvector_view<value_type>(base_view);
}

// option used to specify where factorization takes place
enum class block_factor_location { host, accelerator };

// LAPACK character option casting
inline char to_char(enum class uplo uplo)
{
    switch (uplo)
    {
    case uplo::upper:
        return 'u';
    case uplo::lower:
    default:
        return 'l';
    }
}

inline enum class uplo to_option(const char& uplo)
{
    switch (uplo)
    {
    case 'L':
    case 'l':
        return uplo::lower;
    case 'U':
    case 'u':
    default:
        return uplo::upper;
    }
}

// runtime checks
template <typename value_type, template <typename,int> class container_type>
int require_square(const container_type<value_type,2>& a)
{
    if (a.extent[0] != a.extent[1])
        runtime_error();

    return a.extent[0];
}

template <enum class ordering storage_type, typename value_type, template <typename,int> class container_type>
int get_rows(const container_type<value_type,2>& a)
{
    if (storage_type == ordering::column_major)
        return a.extent[1];
    else
        return a.extent[0];
}

template <enum class ordering storage_type, typename value_type, template <typename,int> class container_type>
int get_cols(const container_type<value_type,2>& a)
{
    if (storage_type == ordering::column_major)
        return a.extent[0];
    else
        return a.extent[1];
}

template <enum class ordering storage_type, typename value_type, template <typename,int> class container_type>
int get_leading_dimension(const container_type<value_type,2>& a)
{
    if (storage_type == ordering::column_major)
        return a.extent[1];
    else
        return a.extent[0];
}

// checks returns from a host LAPACK call
inline void info_check(int info)
{
    if (info > 0)
        data_error(info);
    else if (info < 0)
        argument_error(-info);
}

// a simple host array wrapper
template <unsigned int rank, typename value_type>
class host_array_view
{
public:
    host_array_view(const concurrency::extent<rank>& extent, value_type* ptr)
        : extent(extent), base_ptr(ptr)
    {}

private:
    const concurrency::extent<rank> extent;
    value_type const* base_ptr;
};

// data layout aware sub-matrix extraction method
template <enum class ordering storage_type, typename value_type, template <typename,int> class container_type>
concurrency::array_view<value_type,2> get_sub_matrix(const container_type<value_type,2>& matrix, const concurrency::index<2>& location, const concurrency::extent<2>& size)
{
    if (storage_type == ordering::row_major)
        return matrix.section(location, size);
    else
        return matrix.section(concurrency::index<2>(location[1], location[0]), concurrency::extent<2>(size[1], size[0]));
}

} // namesapce amplapack

#endif AMPLAPACK_RUNTIME_H


