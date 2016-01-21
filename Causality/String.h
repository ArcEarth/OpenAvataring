#pragma once
#include <string>
#include <string_span.h>

namespace Causality
{
	//using gsl::owner;
	using gsl::byte;
	//using gsl::not_null;
	using std::string;
	using std::wstring;
	using gsl::span;
	using gsl::string_span;
	using gsl::wstring_span;
	using gsl::dynamic_range;

	template <typename ValueType, std::ptrdiff_t FirstDimension = gsl::dynamic_range, std::ptrdiff_t... RestDimensions>
	using array_view = gsl::span<ValueType, FirstDimension, RestDimensions...>;

	template <typename ValueType, size_t Rank>
	using strided_array_view = gsl::strided_span<ValueType, Rank>;

	template<std::ptrdiff_t Extent = gsl::dynamic_range>
	using string_view = gsl::string_span<Extent>;
	template<std::ptrdiff_t Extent = gsl::dynamic_range>
	using wstring_view = gsl::wstring_span<Extent>;

	template <typename CharT, std::ptrdiff_t Extent = gsl::dynamic_range>
	using basic_string_view = gsl::basic_string_span<CharT, Extent>;
}