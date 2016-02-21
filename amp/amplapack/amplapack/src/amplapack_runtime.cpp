#include "amplapack_runtime.h"

namespace amplapack {

// exception safe execution wrapper
amplapack_status safe_call_interface(std::function<void(concurrency::accelerator_view& av)>& functor, int& info)
{
    const bool allow_unsafe = false;

    if (allow_unsafe)
    {
        // only used for debugging
        concurrency::accelerator_view av(concurrency::accelerator().default_view);
        functor(av);
        return amplapack_success;
    }

    try
    {
        // for now, simply use the default view
        concurrency::accelerator_view av(concurrency::accelerator().default_view);

        // safely call the amplack routine with the specific view
        functor(av);
    }
    catch(const data_error_exception& e)
    {
        // in traditional LAPACK interface a positive info represents a data error
        // this data error differs from routine to routine, but is typically used to indicate a specific point of failure
        info = e.get();
        return amplapack_data_error;
    }
    catch(const argument_error_exception& e)
    {
        // in traditional LAPACK interfaces a negative info represents a parameter error
        info = -static_cast<int>(e.get());
        return amplapack_argument_error;
    }
    catch(const runtime_error_exception&)
    {
        // this typically indicates an algorithmic error
        return amplapack_internal_error;
    }
    catch (const std::bad_alloc&)
    {
        return amplapack_memory_error;
    }
    catch(const concurrency::runtime_exception& e)
    {
        // return the AMP runtime error code
        info = e.get_error_code();
        return amplapack_runtime_error;
    }
    catch(...)
    {
        // this should not be encountered under normal operation
        return amplapack_unknown_error;
    }

    // no error!
    info = 0;
    return amplapack_success;
}

} // namespace amplapack
