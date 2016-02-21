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
 * AMPBLAS runtime implementation file.
 *
 * This file contains implementation of core runtime routines for the AMPBLAS library. 
 * Mostly, memory management of bound buffers, and error handling routines.
 *
 *---------------------------------------------------------------------------*/
#include <map>
#include <memory>
#include <amp.h>
#include <assert.h>
#include <concurrent_unordered_map.h> // Microsoft specific 
#include "ampcblas_runtime.h"

namespace ampcblas 
{

//----------------------------------------------------------------------------
// ampblas_exception 
//----------------------------------------------------------------------------
ampblas_exception::ampblas_exception(ampblas_result error_code) throw()
    : err_code(error_code) 
{
}

ampblas_exception::ampblas_exception(const char *const& msg, ampblas_result error_code) throw()
    : err_msg(msg), err_code(error_code) 
{
}

ampblas_exception::ampblas_exception(const std::string& msg, ampblas_result error_code) throw()
    : err_msg(msg), err_code(error_code) 
{
}

ampblas_exception::ampblas_exception(const ampblas_exception &other) throw()
    : std::exception(other), err_msg(other.err_msg), err_code(other.err_code) 
{
}

ampblas_exception::~ampblas_exception() throw()
{
}

ampblas_result ampblas_exception::get_error_code() const throw()
{
    return err_code;
}

const char *ampblas_exception::what() const throw()
{
    return  err_msg.data();
}

namespace _details 
{
class amp_buffer;
class thread_context;

namespace 
{
concurrency::critical_section g_allocations_cs;
std::map<const void*, amp_buffer*> g_allocations;

// This is to store per-thread shared resources. Currently two resources last error code 
// and current accelerator_view are stored. concurrent_unordered_map is supported by 
// Microsoft Visual Studio. It is lock-free. You can also use thread_local feature if 
// your development environment supports. 
concurrency::concurrent_unordered_map<DWORD, std::unique_ptr<thread_context>> g_thread_contexts;
}

#define PTR_U64(ptr)      reinterpret_cast<uint64_t>(ptr)

//----------------------------------------------------------------------------
// amp_buffer
//
// adapter class to associate a raw buffer with an array_view 
//----------------------------------------------------------------------------
class amp_buffer
{
public:
	amp_buffer(void *buffer_ptr, size_t byte_len)
		:mem_base(reinterpret_cast<int32_t*>(buffer_ptr)), 
		 byte_length(byte_len),
         // byte_len has been asserted to be multiple of int32_t size, and the byte_len/sizeof(int32_t)
         // is no greater than INT_MAX. So we can safely cast the result to int. 
		 storage(concurrency::extent<1>(static_cast<int>(static_cast<uint64_t>(byte_len)/sizeof(int32_t))), 
                 reinterpret_cast<int32_t*>(buffer_ptr))
	{
    }

    // Check whether this bound buffer contains the region starting at buffer_ptr with length byte_len 
    bool contain(const void *buffer_ptr, size_t byte_len) const
    {
        if (mem_base <= buffer_ptr && (PTR_U64(mem_base)+byte_length >= PTR_U64(buffer_ptr)+byte_len))
        {
            return true;
        }
        return false;
    }

    // Check whether this bound buffer completely excludes the region starting at buffer_ptr with length byte_len
    bool exclusive_with(const void *buffer_ptr, size_t byte_len) const
    {
        if ((PTR_U64(mem_base)+byte_length <= PTR_U64(buffer_ptr)) ||
            (PTR_U64(mem_base) >= PTR_U64(buffer_ptr)+byte_len))
        {
            return true;
        }
        return false;
    }

    // Check whether this bound buffer overlaps with the region starting at buffer_ptr with length byte_len
    // For the case the region is contained in the bound buffer, it returns false. 
    bool overlap(const void *buffer_ptr, size_t byte_len) const
    {
        return !exclusive_with(buffer_ptr, byte_len) && !contain(buffer_ptr, byte_len);
    }
    
	const int32_t *mem_base;
	const size_t byte_length;
	concurrency::array_view<int32_t> storage;
private:
    amp_buffer& operator=(const amp_buffer& right);
};

//----------------------------------------------------------------------------
// thread_context
//
// data structure to be kept in the thread-local-storage for the calling thread
//----------------------------------------------------------------------------

class thread_context
{
public:
	thread_context() : last_error_code(AMPBLAS_OK), curr_accl_view(concurrency::accelerator().default_view)
	{
	}

	ampblas_result last_error_code;
	concurrency::accelerator_view curr_accl_view;
};

//----------------------------------------------------------------------------
// Data management facilities 
//----------------------------------------------------------------------------
#define ASSERT_BUFFER_LENGTH(buf_byte_len) \
    assert(((buf_byte_len) % sizeof(int32_t) == 0) && "Buffer length must be multiple of int32_t size"); \
    assert((static_cast<uint64_t>(buf_byte_len) <= static_cast<uint64_t>(INT_MAX)*sizeof(int32_t)) && "Buffer length overflow");

static inline void check_buffer_length(size_t byte_len)
{
    if (byte_len % sizeof(int32_t) != 0)
    {
        throw ampblas_exception("Buffer length must be multiple of int32_t size", AMPBLAS_BAD_RESOURCE);
    }
    else if (static_cast<uint64_t>(byte_len) > static_cast<uint64_t>(INT_MAX)*sizeof(int32_t))
    {
        throw ampblas_exception("Buffer length overflow", AMPBLAS_BAD_RESOURCE);
    }
}

// Find the bound buffer which contains a region starting at buffer_ptr with length byte_len
//
// returns a bound buffer if the bound buffer contains the buffer [buffer_ptr, buffer_ptr+byte_len)
// returns nullptr if the buffer [buffer_ptr, buffer_ptr+byte_len) is exclusive with any bound buffer 
// throws an ampblas_exception if the buffer [buffer_ptr, buffer_ptr+byte_len) overlaps with another bound buffer
//
// The caller of this function has to hold g_allocations_cs lock for synchronization. 
static amp_buffer* find_amp_buffer(const void *buffer_ptr, size_t byte_len)
{
    ASSERT_BUFFER_LENGTH(byte_len);

    amp_buffer *ampbuff = nullptr;
	amp_buffer *ampbuff_prev = nullptr;

    // No amp_buffer in cache yet. 
	if (g_allocations.size() == 0) 
	{
        return nullptr;
	}

	auto it = g_allocations.lower_bound(const_cast<void*>(buffer_ptr));
    if (it == g_allocations.begin())
    {
        ampbuff = it->second;
	}
    else if (it == g_allocations.end())
    {
        it--; 
        ampbuff = it->second;
	} 
    else
    {
        ampbuff = it->second;
        it--;
        ampbuff_prev = it->second;
	}

    if (ampbuff->contain(buffer_ptr, byte_len)) 
	{
		return ampbuff;
	}
	else if (ampbuff_prev != nullptr && ampbuff_prev->contain(buffer_ptr, byte_len)) 
	{
		return ampbuff_prev;
	}
	else if (ampbuff->overlap(buffer_ptr, byte_len) || 
		ampbuff_prev != nullptr && ampbuff_prev->overlap(buffer_ptr, byte_len)) 
	{
		throw ampblas_exception("Buffer overlapped", AMPBLAS_BAD_RESOURCE);
	}

	return nullptr;
}

concurrency::array_view<int32_t> get_array_view(const void *buffer_ptr, size_t byte_len)
{
    const amp_buffer *ampbuff = nullptr;
    {
		concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);
        ampbuff = find_amp_buffer(buffer_ptr, byte_len);
    }

    if (ampbuff == nullptr) 
    {
        throw ampblas_exception("Unbound resource", AMPBLAS_UNBOUND_RESOURCE);
    }

    check_buffer_length(byte_len);
	int elem_len = static_cast<int>(byte_len / sizeof(int32_t));

    uint64_t byte_offset = PTR_U64(buffer_ptr) - PTR_U64(ampbuff->mem_base);
    if (byte_offset % sizeof(int32_t) != 0)
    {
        throw ampblas_exception("Invalid buffer argument", AMPBLAS_INVALID_ARG);
    }

    uint64_t elem_offset = byte_offset / sizeof(int32_t);
    assert(elem_offset <= INT_MAX);

	return ampbuff->storage.section(concurrency::index<1>(static_cast<int>(elem_offset)), concurrency::extent<1>(elem_len));
}

void bind(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len);
	concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);

    auto ampbuff = find_amp_buffer(buffer_ptr, byte_len);
    if (ampbuff != nullptr)
    {
		throw ampblas_exception("Duplicate binding", AMPBLAS_BAD_RESOURCE);
    }
        
	std::unique_ptr<amp_buffer> buff(new amp_buffer(buffer_ptr, byte_len));
    auto it = g_allocations.insert(std::make_pair(buffer_ptr, buff.get()));
	assert(it.second == true);

    buff.release();
}

bool unbind(void *buffer_ptr)
{
	std::unique_ptr<amp_buffer> ampbuff;
	{
		concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);

        auto it = g_allocations.find(buffer_ptr);
        if (it == g_allocations.end())
        {
            // unbound buffer
            return false;
        }

        ampbuff.reset(it->second);
		g_allocations.erase(it);
	}

    return true;
}

bool ifbound(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len);
	concurrency::critical_section::scoped_lock scope_lock(g_allocations_cs);

    try 
    {
        return (find_amp_buffer(buffer_ptr, byte_len) != nullptr);
    } 
    catch (ampblas_exception&)
    {
        // overlapped buffer
        return false;
    }
}

void synchronize(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len);
	get_array_view(buffer_ptr, byte_len).synchronize();
}

void discard(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len);
	get_array_view(buffer_ptr, byte_len).discard_data();
}

void refresh(void *buffer_ptr, size_t byte_len)
{
    check_buffer_length(byte_len);
    get_array_view(buffer_ptr, byte_len).refresh();
}

thread_context* get_current_thread_context(const DWORD tid)
{
    auto it = g_thread_contexts.find(tid);
    if (it == g_thread_contexts.end())
    {
        auto pair = g_thread_contexts.insert(std::make_pair(tid, std::unique_ptr<thread_context>(new thread_context())));
        assert(pair.second == true);

        it = pair.first;
    }
    return it->second.get();
}

// Sets the last-error code for the calling thread
void set_last_error(const ampblas_result error_code)
{
    auto curr_context = get_current_thread_context(GetCurrentThreadId());
    assert(curr_context != nullptr);

    curr_context->last_error_code = error_code;
}

ampblas_result get_last_error()
{
    auto curr_context = get_current_thread_context(GetCurrentThreadId());
    assert(curr_context != nullptr);

    return curr_context->last_error_code;
}

} // namespace _details


// The current accelerator_view is kept in thread-local-storage for the calling thread
void set_current_accelerator_view(const concurrency::accelerator_view& accl_view)
{
    auto curr_context = _details::get_current_thread_context(GetCurrentThreadId());
    assert(curr_context != nullptr);

	curr_context->curr_accl_view = accl_view;
}

// The accelerator_view associated with the default accelerator is returned, if no
// set_current_accelerator_view() is called previously.
concurrency::accelerator_view get_current_accelerator_view()
{
    auto curr_context = _details::get_current_thread_context(GetCurrentThreadId());
    assert(curr_context != nullptr);

	return curr_context->curr_accl_view;
}

} // namespace ampcblas

//----------------------------------------------------------------------------
// AMPBLAS runtime facilities for C BLAS 
//----------------------------------------------------------------------------
#define AMPCBLAS_RUNTIME_CHECKED_CALL(expr) \
    { \
        ampblas_result re = AMPBLAS_OK; \
        try \
        { \
            (expr);\
        } \
        catch (ampcblas::ampblas_exception &e) \
        { \
            re = e.get_error_code(); \
        } \
        catch (concurrency::runtime_exception&) \
        { \
            re = AMPBLAS_AMP_RUNTIME_ERROR; \
        } \
        catch (std::bad_alloc&) \
        { \
            re = AMPBLAS_OUT_OF_MEMORY; \
        } \
        catch (...) \
        { \
            re = AMPBLAS_INTERNAL_ERROR; \
        } \
        ampblas_set_last_error(re); \
        return re; \
    }

extern "C" ampblas_result ampblas_bind(void *buffer_ptr, size_t byte_len)
{
    AMPCBLAS_RUNTIME_CHECKED_CALL(ampcblas::_details::bind(buffer_ptr, byte_len));
}

extern "C" bool ampblas_unbind(void *buffer_ptr)
{
    return ampcblas::_details::unbind(buffer_ptr);
}

extern "C" bool ampblas_ifbound(void *buffer_ptr, size_t byte_len)
{
    return ampcblas::_details::ifbound(buffer_ptr, byte_len);
}

extern "C" ampblas_result ampblas_synchronize(void *buffer_ptr, size_t byte_len)
{
    AMPCBLAS_RUNTIME_CHECKED_CALL(ampcblas::_details::synchronize(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_discard(void *buffer_ptr, size_t byte_len)
{
	AMPCBLAS_RUNTIME_CHECKED_CALL(ampcblas::_details::discard(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_refresh(void *buffer_ptr, size_t byte_len)
{
    AMPCBLAS_RUNTIME_CHECKED_CALL(ampcblas::_details::refresh(buffer_ptr, byte_len));
}

extern "C" ampblas_result ampblas_set_current_accelerator_view(void *accl_view)
{
    if (accl_view == nullptr)
        return AMPBLAS_INVALID_ARG;

    const concurrency::accelerator_view& av = *reinterpret_cast<concurrency::accelerator_view*>(accl_view);
	AMPCBLAS_RUNTIME_CHECKED_CALL(ampcblas::set_current_accelerator_view(av));
}

extern "C" ampblas_result ampblas_get_last_error()
{
    return ampcblas::_details::get_last_error();
}

extern "C" void ampblas_set_last_error(const ampblas_result error_code)
{
    return ampcblas::_details::set_last_error(error_code);
}

extern "C" void ampblas_xerbla(const char *srname, int * info)
{
	printf("** On entry to %6s, parameter number %2i had an illegal value\n", srname, *info);
}

