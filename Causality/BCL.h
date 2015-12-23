#pragma once

// STL and std libs
#include <map>
#include <set>
#include <vector>
#include <list>
#include <memory>
#include <functional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <type_traits>

// Guildline Support Library
#include <gsl.h>

//// boost and std extension
//#include <boost\signals2.hpp>
//#include <boost\any.hpp>
#include <boost\range\iterator_range_core.hpp>
#include <boost\range\adaptor\transformed.hpp>
//#include <boost\operators.hpp>
//#include <boost\format.hpp>
//#include <boost\filesystem.hpp>

#include "Common\stride_range.h"
#include "Common\tree.h"

#include "Math3D.h"
#include "SmartPointers.h"
#include "String.h"

#if defined __AVX__
#undef __AVX__ //#error Eigen have problem with AVX now
#endif
#define EIGEN_HAS_CXX11_MATH 1
#define EIGEN_HAS_STD_RESULT_OF 1
#define EIGEN_HAS_VARIADIC_TEMPLATES 1
#include <Eigen\Dense>


namespace Causality
{
	using time_seconds = std::chrono::duration<double>;
	typedef uint64_t id_t;

	using std::string;
	using boost::iterator_range;
	//using boost::sub_range;

	//namespace adaptors = boost::adaptors;

	using gsl::owner;
	using gsl::byte;
	using gsl::not_null;

	using stdx::tree_node;
	using stdx::foward_tree_node;
	using stdx::stride_range;
	using stdx::stride_iterator;

	using std::vector;
	using std::map;
	using std::function;
	using std::unique_ptr;
	using std::shared_ptr;
	using std::list;
	using std::weak_ptr;
}