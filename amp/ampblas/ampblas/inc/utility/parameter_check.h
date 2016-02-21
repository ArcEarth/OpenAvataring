#ifndef AMPBLAS_UTILITY_PARAMETER_CHECK_H
#define AMPBLAS_UTILITY_PARAMETER_CHECK_H

#include <string>

#include "ampblas_config.h"

AMPBLAS_NAMESPACE_BEGIN
DETAIL_NAMESPACE_BEGIN

//
// Helper Routines
// 

inline bool is_equal(const concurrency::extent<1>& a, const concurrency::extent<1>& b) restrict(amp,cpu)
{
    return (a[0] == b[0]);
}

inline bool is_equal(const concurrency::extent<2>& a, const concurrency::extent<2>& b) restrict(amp,cpu)
{
    return (a[0] == b[0] && a[1] == b[1]);
}

inline bool is_equal(const concurrency::extent<3>& a, const concurrency::extent<3>& b) restrict(amp,cpu)
{
    return (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]);
}

inline bool is_square(const concurrency::extent<1>& /*a*/) restrict(amp,cpu)
{
    return false;
}

inline bool is_square(const concurrency::extent<2>& a) restrict(amp,cpu)
{
    return (a[0] == a[1]);
}

inline bool is_square(const concurrency::extent<3>& a) restrict(amp,cpu)
{
    return (a[0] == a[1] == a[2]);
}

//
// Generic Routines
//

template <enum class order S, unsigned int rank>
void require_square(const std::string a_name, const concurrency::extent<rank>& a_extent)
{
    if (!is_square(a_extent))
    {
        std::string message = "the array " + a_name + " must be square";
        argument_error(message);
    }
}

template <enum class order S, unsigned int rank>
void require_same_dimensions(const std::string a_name, const concurrency::extent<rank>& a_extent, const std::string b_name, const concurrency::extent<rank>& b_extent)
{
    if (!is_equal(a_extent, b_extent))
    {
        std::string message = "the dimensions of " + a_name + " and " + b_name + " must match"; 
        argument_error(message);
    }
}

template <enum class order S>
void require_equal_rows(const std::string a_name, const concurrency::extent<2>& a_extent, const std::string b_name, const concurrency::extent<2>& b_extent)
{
    if (rows<S>(a_extent) != rows<S>(b_extent))
    {
        std::string message = "the row count of " + a_name + " and " + b_name + " must match";
        argument_error(message);
    }
}

template <enum class order S>
void require_equal_columns(const std::string a_name, const concurrency::extent<2>& a_extent, const std::string b_name, const concurrency::extent<2>& b_extent)
{
    if (columns<S>(a_extent) != columns<S>(b_extent))
    {
        std::string message = "the column count of " + a_name + " and " + b_name + " must match";
        argument_error(message);
    }
}

template <enum class order S>
void require_equal_outer(const std::string a_name, const concurrency::extent<2>& a_extent, const std::string b_name, const concurrency::extent<2>& b_extent)
{
    if (row<S>(a_extent) != columns<S>(b_extent))
    {
        std::string message = "the outer dimensions of " + a_name + " and " + b_name + " must match";
        argument_error(message);
    }
}

template <enum class order S>
void require_equal_inner(const std::string a_name, const concurrency::extent<2>& a_extent, std::string b_name, const concurrency::extent<2>& b_extent)
{
    if (columns<S>(a_extent) != rows<S>(b_extent))
    {
        std::string message = "the inner dimensions of " + a_name + " and " + b_name + " must match";
        argument_error(message);
    }
}

//
// Helper Macros
// 

#define AMPBLAS_STRINGIZE(X) AMPBLAS_DO_STRINGIZE(X)
#define AMPBLAS_DO_STRINGIZE(X) #X

#define AMPBLAS_REQUIRE_SQUARE(S,A) _detail::require_square<S>(AMPBLAS_STRINGIZE(A), A.extent)
#define AMPBLAS_REQUIRE_EQUAL(S,A,B) _detail::require_equal<S>(AMPBLAS_STRINGIZE(A), A.extent, AMPBLAS_STRINGIZE(B), B.extent)
#define AMPBLAS_REQUIRE_EQUAL_ROWS(S,A,B) _detail::require_equal_rows<S>(AMPBLAS_STRINGIZE(A), A.extent, AMPBLAS_STRINGIZE(B), B.extent)
#define AMPBLAS_REQUIRE_EQUAL_COLUMNS(S,A,B) _detail::require_equal_columns<S>(AMPBLAS_STRINGIZE(A), A.extent, AMPBLAS_STRINGIZE(B), B.extent)
#define AMPBLAS_REQUIRE_EQUAL_OUTER(S,A,B) _detail::require_equal_outer<S>(AMPBLAS_STRINGIZE(A), A.extent, AMPBLAS_STRINGIZE(B), B.extent)
#define AMPBLAS_REQUIRE_EQUAL_INNER(S,A,B) _detail::require_equal_inner<S>(AMPBLAS_STRINGIZE(A), A.extent, AMPBLAS_STRINGIZE(B), B.extent)

DETAIL_NAMESPACE_END
AMPBLAS_NAMESPACE_END

#endif // AMPBLAS_UTILITY_PARAMETER_CHECK_H
