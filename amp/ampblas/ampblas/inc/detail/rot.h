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
 * rot.h
 *
 *---------------------------------------------------------------------------*/

#include "ampblas_dev.h"

namespace ampblas {

//-------------------------------------------------------------------------
// ROT
//-------------------------------------------------------------------------

template <typename x_type, typename y_type, typename c_type, typename s_type>
void rot(const concurrency::accelerator_view& av, x_type&& x, y_type&& y, c_type&& c, s_type&& s)
{
    concurrency::parallel_for_each(
        av, 
        x.extent, 
        [=] (concurrency::index<1> idx) restrict(amp) 
        {
            auto temp = c * x[idx] + s * y[idx];
            y[idx] = c * y[idx] - s * x[idx];
            x[idx] = temp;
        }
    );
}

} // namespace ampblas
