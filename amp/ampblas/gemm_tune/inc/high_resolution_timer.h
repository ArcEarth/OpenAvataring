// high resolution timer 

#ifndef AMBLAS_GEMM_PROFILE_HIGH_RESOLUTION_TIMER_H
#define AMBLAS_GEMM_PROFILE_HIGH_RESOLUTION_TIMER_H

#include <memory>

#include <tune.h>

TUNE_NAMESPACE_BEGIN

class high_resolution_timer
{
public:
    high_resolution_timer();
    ~high_resolution_timer();

    void restart();
    double elapsed();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

TUNE_NAMESPACE_END

#endif // AMBLAS_GEMM_PROFILE_HIGH_RESOLUTION_TIMER_H
