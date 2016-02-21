#include <windows.h>

#include "amplapack_test.h"

struct high_resolution_timer::impl
{
    LARGE_INTEGER start_time;
};

high_resolution_timer::high_resolution_timer()
{
    pimpl = std::unique_ptr<impl>(new impl);
    restart();
}

high_resolution_timer::~high_resolution_timer()
{
    // default
}

void high_resolution_timer::restart()
{
    QueryPerformanceCounter(&pimpl->start_time); 
}

double high_resolution_timer::elapsed()
{
    LARGE_INTEGER end_time;
    QueryPerformanceCounter(&end_time);
    LARGE_INTEGER PerformanceFrequency_hz;
    QueryPerformanceFrequency(&PerformanceFrequency_hz);
    double frequency = double(PerformanceFrequency_hz.QuadPart);
    return double(end_time.QuadPart - pimpl->start_time.QuadPart) / frequency;
}
