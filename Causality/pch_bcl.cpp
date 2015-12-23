#define _PCH_CPP_
//#ifdef EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TYPE, SIZE)
//#undef EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TYPE, SIZE)
//#define EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TYPE, SIZE) 
//#endif

#include "pch_bcl.h"

template class std::vector<int>;
template class std::vector<float>;
//template class Eigen::Matrix<float, -1, -1>;
//template class Eigen::Matrix<float, -1,  1>;
//template class Eigen::Matrix<float,  1, -1>;

