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
 * ampcblas_config.h
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPCBLAS_CONFIG_H
#define AMPCBLAS_CONFIG_H

// interface with ampblas complex types and options
#include "ampblas_complex.h"
#include "ampblas_defs.h"

#include "ampcblas.h"
#include "ampcblas_runtime.h"
#include "ampcblas_complex.h"

namespace ampcblas {

// ampblas complex types
typedef ampblas::complex<float>  fcomplex;
typedef ampblas::complex<double> dcomplex;

// ampcblas -> ampblas complex casts
inline const fcomplex* ampblas_cast(const ampblas_fcomplex* ptr)
{
    return reinterpret_cast<const fcomplex*>(ptr);
}

inline fcomplex* ampblas_cast(ampblas_fcomplex* ptr)
{
    return reinterpret_cast<fcomplex*>(ptr);
}

inline const dcomplex* ampblas_cast(const ampblas_dcomplex* ptr)
{
    return reinterpret_cast<const dcomplex*>(ptr);
}

inline dcomplex* ampblas_cast(ampblas_dcomplex* ptr)
{
    return reinterpret_cast<dcomplex*>(ptr);
}

// ampcblas option -> ampblas options
inline enum class ampblas::transpose cast(const enum AMPBLAS_TRANSPOSE& trans)
{
    switch (trans)
    {
    case AmpblasNoTrans:
        return ampblas::transpose::no_trans;
    case AmpblasTrans:
        return ampblas::transpose::trans;
    case AmpblasConjTrans:
    default:
        return ampblas::transpose::conj_trans;
    }
}

inline enum class ampblas::diag cast(const enum AMPBLAS_DIAG& diag)
{
    switch (diag)
    {
    case AmpblasUnit:
        return ampblas::diag::unit;
    case AmpblasNonUnit:
    default:
        return ampblas::diag::non_unit;
    }
}

inline enum class ampblas::side cast(const enum AMPBLAS_SIDE& side)
{
    switch (side)
    {
    case AmpblasLeft:
        return ampblas::side::left;
    case AmpblasRight:
    default:
        return ampblas::side::right;
    }
}

inline enum class ampblas::uplo cast(const enum AMPBLAS_UPLO& uplo)
{
    switch (uplo)
    {
    case AmpblasUpper:
        return ampblas::uplo::upper;
    case AmpblasLower:
    default:
        return ampblas::uplo::lower;
    }
}


//---------------------------------------------------------------------------- 
// Exceptions
//----------------------------------------------------------------------------

class argument_error_exception
{
public:
    argument_error_exception(const std::string& name, int index)
        : name_(name), index_(index)
    {}

    std::string name() const
    {
        return name_;
    }

    unsigned int index() const
    {
        return index_;
    }

private:
    std::string name_;
    unsigned int index_;
};


class feature_not_implemented_exception {};

//---------------------------------------------------------------------------- 
// Exception Throwers
//----------------------------------------------------------------------------

inline void argument_error(const std::string& name, int index)
{
    throw argument_error_exception(name, index);
}

inline void feature_not_implemented()
{
    throw feature_not_implemented_exception();
}

 } // namespace ampcblas 

#define AMPBLAS_CHECKED_CALL(...)                        \
{                                                        \
    ampblas_result re = AMPBLAS_OK;                      \
    try                                                  \
    {                                                    \
        (__VA_ARGS__);                                   \
    }                                                    \
    catch (const ampcblas::argument_error_exception& e)  \
    {                                                    \
        int index = int(e.index());                      \
        ampblas_xerbla(e.name().c_str(), &index);        \
        re = AMPBLAS_INVALID_ARG;                        \
    }                                                    \
    catch (ampcblas::feature_not_implemented_exception&) \
    {                                                    \
        re = AMPBLAS_NOT_SUPPORTED_FEATURE;              \
    }                                                    \
    catch (concurrency::runtime_exception&)              \
    {                                                    \
        re = AMPBLAS_AMP_RUNTIME_ERROR;                  \
    }                                                    \
    catch (std::bad_alloc&)                              \
    {                                                    \
        re = AMPBLAS_OUT_OF_MEMORY;                      \
    }                                                    \
    catch (...)                                          \
    {                                                    \
        re = AMPBLAS_INTERNAL_ERROR;                     \
    }                                                    \
    ampblas_set_last_error(re);                          \
}

#endif // AMPCBLAS_CONFIG_H
