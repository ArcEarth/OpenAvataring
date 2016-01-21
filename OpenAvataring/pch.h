#include "Causality\pch_bcl.h"

//#if defined __AVX__
//#undef __AVX__ //#error Eigen have problem with AVX now
//#endif
#define EIGEN_HAS_CXX11_MATH 1
#define EIGEN_HAS_STD_RESULT_OF 1
#define EIGEN_HAS_VARIADIC_TEMPLATES 1
#include <Eigen\Dense>