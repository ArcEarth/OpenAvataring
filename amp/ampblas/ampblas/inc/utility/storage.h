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
 * storage.h 
 *
 * Utility routines related to columnn/row major storage concerns
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_UTILITY_STORAGE_H
#define AMPBLAS_UTILITY_STORAGE_H

#include "ampblas_config.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// TODO: use this enum instead of the one in the AMPBLAS namespace
// enum class storage { row_major, col_major };

// storage-specific index helper
template <order S> 
inline concurrency::index<2> index(unsigned int i, unsigned int j);

template <>
inline concurrency::index<2> index<order::row_major>(unsigned int i, unsigned int j)
{ 
    return concurrency::index<2>(i,j);
}

template <> 
inline concurrency::index<2> index<order::col_major>(unsigned int i, unsigned int j) 
{ 
    return concurrency::index<2>(j,i);
}

// storage-specific extent helper
template <order S> 
inline concurrency::extent<2> extent(unsigned int i, unsigned int j);

template <>
inline concurrency::extent<2> extent<order::row_major>(unsigned int i, unsigned int j)
{ 
    return concurrency::extent<2>(i,j);
}

template <> 
inline concurrency::extent<2> extent<order::col_major>(unsigned int i, unsigned int j) 
{ 
    return concurrency::extent<2>(j,i);
}

// storage-specfic row count extaction
template <enum class order S>
inline unsigned int rows(const concurrency::extent<2>& extent);

template <>
inline unsigned int rows<order::row_major>(const concurrency::extent<2>& extent)
{
    return extent[0];
}

template <>
inline unsigned int rows<order::col_major>(const concurrency::extent<2>& extent)
{
    return extent[1];
}

// storage-specfic column count extaction
template <enum class order S>
inline unsigned int columns(const concurrency::extent<2>& extent);

template <>
inline unsigned int columns<order::row_major>(const concurrency::extent<2>& extent)
{
    return extent[1];
}

template <>
inline unsigned int columns<order::col_major>(const concurrency::extent<2>& extent)
{
    return extent[0];
}

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_UTILITY_STORAGE_H