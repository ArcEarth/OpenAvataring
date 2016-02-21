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
 *---------------------------------------------------------------------------*/

#include <windows.h>

#include "ampblas_test_timer.h"

struct high_resolution_timer::impl
{
	impl() {}
    LARGE_INTEGER start_time;
};

high_resolution_timer::high_resolution_timer()
{
    pimpl = std::make_shared<impl>();
    restart();
}

void high_resolution_timer::restart()
{
    QueryPerformanceCounter(&pimpl->start_time); 
}

// default elapsed timer
double high_resolution_timer::elapsed() const
{
    LARGE_INTEGER end_time;
    QueryPerformanceCounter(&end_time);
    LARGE_INTEGER PerformanceFrequency_hz;
    QueryPerformanceFrequency(&PerformanceFrequency_hz);
    double frequency = static_cast<double>(PerformanceFrequency_hz.QuadPart);
    double time_seconds = static_cast<double>(end_time.QuadPart - pimpl->start_time.QuadPart)/frequency;
    return time_seconds;
}

// different resolutions
double high_resolution_timer::s() const
{
    return elapsed();
}

double high_resolution_timer::ms() const
{
    return elapsed() * double(1e3);
}

double high_resolution_timer::us() const
{
    return elapsed() * double(1e6);
}

double high_resolution_timer::ns() const
{
    return elapsed() * double(1e9);
}
