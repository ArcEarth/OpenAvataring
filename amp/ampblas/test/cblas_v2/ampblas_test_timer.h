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
 * ampblas_test_timer.h
 *
 * High resolution timer for Windows systems.
 *
 *---------------------------------------------------------------------------*/

#pragma once

#include <memory>

// as of VS11, std::chrono::high_resolution_clock has a resolution of a few ms
// this is currently too large for proper kernel testing

class high_resolution_timer
{
public:

    high_resolution_timer();
    void restart();

    // default elapsed timer
    double elapsed() const;

    // different resolutions
    double s() const;
    double ms() const;
    double us() const;
    double ns() const;

private:

    // firewall to keep OS details out of header
    struct impl;
    std::shared_ptr<impl> pimpl;
};
