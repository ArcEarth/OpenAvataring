#ifndef AMPBLAS_TUNE_H
#define AMPBLAS_TUNE_H

#include "ampblas_dev.h"

#include <string>

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

// a list of architectures used by the tuning framework
enum class architecture
{ 
    // AMD architectures
    amd,                        // generic

    // NVIDIA architectures 
    nvidia,                     // generic
    nvidia_xxx,                 // unused example of another architecture that could be added

    // fall back
    unknown 
};

// incredibly simple architecture parser
inline enum class architecture get_architecture(std::wstring& description)
{
    const std::wstring& amd_key(L"AMD");
    const std::wstring& nvidia_key(L"NVIDIA");

    if (description.find(amd_key) != std::string::npos)
        return architecture::amd;
    else if (description.find(nvidia_key) != std::string::npos)
        return architecture::nvidia;
    else
        return architecture::unknown;
    
}

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_TUNE_H
