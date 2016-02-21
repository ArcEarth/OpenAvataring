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
 * typedefs for AMPBLAS.
 *
 * This file contains common typedefs for AMP C++ BLAS.
 *
 *---------------------------------------------------------------------------*/
#ifndef AMPBLAS_DEFS_H
#define AMPBLAS_DEFS_H

#include <string>

namespace ampblas {

// options
enum class order { row_major, col_major };
enum class transpose { no_trans, trans, conj_trans };
enum class uplo { upper, lower };
enum class diag { non_unit, unit };
enum class side { left, right };

// argument exception
class argument_error_exception
{
public:
    argument_error_exception()
        : description_("argument error")
    {}

    argument_error_exception(const std::string& description)
        : description_(description)
    {}
    
    std::string what() const
    {
        return description_;
    }

private:
    std::string description_;
};

inline void argument_error(const std::string& description)
{ 
    throw argument_error_exception(description); 
}

} // namespace ampblas

#endif // AMPBLAS_DEFS_H
