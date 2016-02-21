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
 * copy.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {

//-------------------------------------------------------------------------
// COPY
//   copy a container to another container. The two containers cannot be 
//   overlpped.
//-------------------------------------------------------------------------

template <typename x_type, typename y_type>
void copy(const concurrency::accelerator_view& av, const x_type& x, y_type& y)
{
    _detail::copy(av, x.extent, x, y);
}

} // namespace ampblas
