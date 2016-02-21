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
 * AMP CBLAS and AMP C++ BLAS library runtime header.
 *
 * This file contains APIs of data management and data transformation of bound buffer,
 * and APIs of accelerator_view selection and error handling. 
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPCBLAS_RUNTIME_H
#define AMPCBLAS_RUNTIME_H

#include "ampcblas_defs.h"

//----------------------------------------------------------------------------
// AMPBLAS runtime for C++ BLAS 
//----------------------------------------------------------------------------
#ifdef __cplusplus
#include <exception>
#include <string>
#include <amp.h>

namespace ampcblas 
{ 
//----------------------------------------------------------------------------
// ampblas_exception 
//----------------------------------------------------------------------------
class ampblas_exception : public std::exception
{
public:
AMPBLAS_DLL ampblas_exception(const char *const& msg, ampblas_result error_code) throw();
AMPBLAS_DLL ampblas_exception(const std::string& msg, ampblas_result error_code) throw();
AMPBLAS_DLL explicit ampblas_exception(ampblas_result error_code) throw();
AMPBLAS_DLL ampblas_exception(const ampblas_exception &other) throw();
AMPBLAS_DLL virtual ~ampblas_exception() throw();
AMPBLAS_DLL ampblas_result get_error_code() const throw();
AMPBLAS_DLL virtual const char *what() const throw();

private:
    ampblas_exception &operator=(const ampblas_exception &);
    std::string err_msg;
    ampblas_result err_code;
};

//----------------------------------------------------------------------------
//
// Data management APIs
//
// bind specifies a region of host memory which should become available
// to manipulation by AMPBLAS routines. A memory region should not be bound more
// than once. Also, binding of overlapping regions is not supported.
//
// To remove a binding call unbind. This function should only be invoked
// on the exact regions which were previously bound. Otherwise the function returns
// false.
//
// Bindings are (could be) implemented using array_view and they expose a similar
// contract for sychronizing data to the main copy, discarding current changes,
// and refreshing the contents of the binding after it has been modified directly
// in host memory. The functions synchronize, discard, and 
// refresh, respectively, serve these purposes.
//
// The byte length of the bound buffer needs to be multiple of 4 bytes.
// 
// TODO: consider allowing multiple and concurrent bindings of the same buffer
//
//----------------------------------------------------------------------------
namespace _details
{
AMPBLAS_DLL void bind(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL bool unbind(void *buffer_ptr);
AMPBLAS_DLL bool ifbound(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL void synchronize(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL void discard(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL void refresh(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL concurrency::array_view<int32_t> get_array_view(const void *buffer_ptr, size_t byte_len);
} // nampespace _details

template<typename T> 
inline void bind(T *buffer_ptr, size_t element_count)
{
    _details::bind(buffer_ptr, element_count * sizeof(T));
}

template<typename T> 
inline bool unbind(T *buffer_ptr)
{
    return _details::unbind(buffer_ptr);
}

template<typename T> 
inline void synchronize(T *buffer_ptr, size_t element_count)
{
    _details::synchronize(buffer_ptr, element_count * sizeof(T));
}

template<typename T> 
inline void discard(T *buffer_ptr, size_t element_count)
{
    _details::discard(buffer_ptr, element_count * sizeof(T));
}

template<typename T> 
inline void refresh(T *buffer_ptr, size_t element_count)
{
    _details::refresh(buffer_ptr, element_count * sizeof(T));
}

// This API allows binding a single dimensional array_view as an AMPBLAS pointer. It can
// be used to import GPU-based arrays or staging arrays into the AMPBLAS sandbox.
// TODO: not supported yet.
template <typename value_type>
void* bind_array_view(const concurrency::array_view<value_type>& av);

// Conversely, a bound buffer can be obtained and manipulated as an array view.
template<typename value_type>
inline concurrency::array_view<value_type> get_array_view(const value_type *ptr, size_t element_count)
{
	auto av = _details::get_array_view(ptr, element_count * sizeof(value_type));
	return av.reinterpret_as<value_type>();
}

// set_current_accelerator_view set the accelerator view which will be used
// in subsequent AMPBLAS calls. There is no restriction on data access---once data is bound
// using bind, it could be used on any accelerator.
// 
// Current accelerator_view is per-thread basis. In the scenerios where threads can be
// reclaimed, such as in thread-pool model, the reclaimed thread can pick up the accelerator_view 
// set previously, if it has not been reset.  
AMPBLAS_DLL void set_current_accelerator_view(const concurrency::accelerator_view& acc_view);

// Returns the accelerator_view set in last set_current_accelerator_view call. If 
// set_current_accelerator_view has never been called in current thread, the default accelerator_view 
// associated with the default accelerator will be used. 
AMPBLAS_DLL concurrency::accelerator_view get_current_accelerator_view();

//----------------------------------------------------------------------------
// Data transformation operators and utility APIs
//----------------------------------------------------------------------------
template <int rank>
inline concurrency::index<rank> index_scalar(int s) restrict(cpu, amp)
{
	concurrency::index<rank> idx;
	for (int i=0; i<rank; i++)
		idx[i] = s;
	return idx;
}

template <int rank>
inline concurrency::index<rank> index_unity() restrict(cpu, amp)
{
	return index_scalar<rank>(1);
}

template <int rank>
inline concurrency::index<rank> last_index_of(const concurrency::extent<rank> &e) restrict(cpu, amp)
{
	concurrency::index<rank> idx;
	for (int i=0; i<rank; i++)
		idx[i] = e[i]-1;
	return idx;
}

inline concurrency::extent<1> make_extent(int n) restrict(cpu, amp)
{
	return concurrency::extent<1>(n);
}

inline concurrency::extent<2> make_extent(int m, int n) restrict(cpu, amp)
{
	return concurrency::extent<2>(n,m);
}

//----------------------------------------------------------------------------
// stride_view
//
// template class stride_view wraps over an existing array-like class and 
// accesses the underlying storage with a stride.
//
//----------------------------------------------------------------------------
template <typename base_view_type>
class stride_view
{
public:
	typedef typename base_view_type::value_type value_type;
	static const int rank = base_view_type::rank;

    // match interface of array_view
	concurrency::extent<rank> extent;

	// The stride provided may be negative, in which case elements are retrieved in reverse order.
	stride_view(const base_view_type& bv, int stride, const concurrency::extent<rank>& logical_extent) restrict(cpu,amp)
		:base_view(bv), 
		 stride(stride),
         extent(logical_extent),
		 base_index(stride >= 0 ? concurrency::index<rank>() : -stride * last_index_of(logical_extent))
	{
	}

	~stride_view() restrict(cpu,amp) {}

	value_type& operator[] (const concurrency::index<1>& idx) const restrict(cpu,amp)
	{
		return base_view[base_index + stride * idx];
	}

    const base_view_type& get_base_view() const
    {
        return base_view;
    }

private:
    stride_view& operator=(const stride_view& rhs);

	const int stride;
	const concurrency::index<rank> base_index;
    base_view_type base_view;
};

template <typename base_view_type>
inline stride_view<base_view_type> make_stride_view(const base_view_type& bv, int stride, const concurrency::extent<base_view_type::rank>& logical_extent)
{
	return stride_view<base_view_type>(bv, stride, logical_extent);
}

template <typename value_type>
inline stride_view<concurrency::array_view<value_type,1>> make_vector_view(int N, value_type *X, int incX)
{
    auto avX = get_array_view(X, N*std::abs(incX));
    return make_stride_view(avX, incX, make_extent(N));
}


template <typename value_type>
inline stride_view<concurrency::array_view<value_type,1>> make_vector_view(int N, const value_type *X, int incX)
{
    // TODO: there are a number of const-correct issues in the 1d reduction code
    auto avX = get_array_view(const_cast<value_type*>(X), N*std::abs(incX));
    return make_stride_view(avX, incX, make_extent(N));
}

template <typename value_type>
inline concurrency::array_view<value_type,2> make_matrix_view(int M, int N, value_type *A, int ldA) 
{
    return get_array_view(A, ldA*N).view_as(concurrency::extent<2>(N,ldA)).section(concurrency::extent<2>(N,M));
}

template <typename value_type>
inline concurrency::array_view<const value_type,2> make_matrix_view(int M, int N, const value_type *A, int ldA) 
{
    return get_array_view(A, ldA*N).view_as(concurrency::extent<2>(N,ldA)).section(concurrency::extent<2>(N,M));
}

} // namespace ampblas
#endif // __cplusplus

//----------------------------------------------------------------------------
// AMPBLAS runtime for C BLAS 
//----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
//
// Data management API's 
// These are adapters to call the AMP C++ BLAS runtime data management functions. 
//
AMPBLAS_DLL ampblas_result ampblas_bind(void *buffer_ptr, size_t byte_len);

// This function doesn't change the last_error_code
AMPBLAS_DLL bool           ampblas_unbind(void *buffer_ptr);

// This function doesn't change the last_error_code
AMPBLAS_DLL bool           ampblas_ifbound(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL ampblas_result ampblas_synchronize(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL ampblas_result ampblas_discard(void *buffer_ptr, size_t byte_len);
AMPBLAS_DLL ampblas_result ampblas_refresh(void *buffer_ptr, size_t byte_len);

// ampblas_set_current_accelerator_view set the accelerator view which will be used
// in subsequent AMPBLAS calls. If this function has not been called in current 
// thread, the default accelerator_view associated with the default accelerator 
// will be used. 
//
// There is no restriction on data access---once data is bound using ampblas_bind, 
// it could be used on any accelerator.
// 
// Current accelerator_view is per-thread basis. In the scenerios where threads can be
// reclaimed, such as in thread-pool model, the reclaimed thread can pick up the accelerator_view 
// set previously, if it has not been reset.  
//
// returns AMPBLAS_INVALID_ARG if the accl_view argument is nullptr
AMPBLAS_DLL ampblas_result ampblas_set_current_accelerator_view(void * accl_view);

// Retrieves the calling thread's last-error code value. The last-error code is maintained 
// on a per-thread basis. Multiple threads do not overwrite each other's last-error code.
// You should call this function immediately when you want to check the status of a blas 
// function call. That is because some functions set the last-error code with AMPBLAS_OK 
// when they succeed, wiping out the error code set by the most recently failed function.
AMPBLAS_DLL ampblas_result ampblas_get_last_error();

// Sets the last-error code for the calling thread
AMPBLAS_DLL void ampblas_set_last_error(const ampblas_result error_code);

AMPBLAS_DLL void ampblas_xerbla(const char *srname, int * info);

#ifdef __cplusplus
}
#endif

#endif //AMPCBLAS_RUNTIME_H

