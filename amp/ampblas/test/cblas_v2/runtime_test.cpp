#include <amp.h>
#include <vector>
#include <iostream>

#include "ampblas_test.h"
#include "ampblas_test_util.h"
#include "ampblas_runtime.h"

using namespace concurrency;

void runtime_test()
{
    std::vector<accelerator> accs = accelerator::get_all(); 
    std::for_each(accs.begin(), accs.end(), [] (accelerator acc) 
    { 
        std::wcout << "Accelerator: " << acc.description << std::endl; 
        std::wcout << "  device_path = " << acc.device_path << std::endl; 
        std::wcout << "  version = " << (acc.version >> 16) << '.' << (acc.version & 0xFFFF) << std::endl; 
        std::wcout << "  dedicated_memory = " << acc.dedicated_memory << " KB" << std::endl; 
        std::wcout << "  doubles = " << ((acc.supports_double_precision) ? "true" : "false") << std::endl; 
        std::wcout << "  limited_doubles = " << ((acc.supports_limited_double_precision) ? "true" : "false") << std::endl; 
        std::wcout << "  has_display = " << ((acc.has_display) ? "true" : "false") << std::endl;
        std::wcout << "  is_emulated = " << ((acc.is_emulated) ? "true" : "false") << std::endl; 
        std::wcout << "  is_debug = " << ((acc.is_debug) ? "true" : "false") << std::endl; 
        std::cout << std::endl; 
    });
}
