#pragma once

#include <amp_short_vectors.h>

namespace Concurrency
{
	namespace graphics
	{
		template <typename _T, int _Rows, int _Cols>
		struct short_matrix
		{
			typedef _T value_type;
			constexpr static int rows = _Rows;
			constexpr static int cols = _Cols;

			static_assert(rows > 1 && rows <= 4 && cols > 1 && cols <= 4, "short matrix dimension must within be (1,4]*(1,4]");

			using row_vector_t = typename short_vector<value_type, cols>::type;

			row_vector_t r[rows];

			inline row_vector_t operator[](int idx) const __GPU
			{
				return r[idx];
			}

			inline row_vector_t& operator[](int idx) __GPU
			{
				return r[idx];
			}
		};

		template <int Rows>
		using float_Xx4 = short_matrix<float, Rows, 4>;

		using float_1x4 = float_4;
		using float_2x4 = short_matrix<float, 2, 4>;
		using float_3x4 = short_matrix<float, 3, 4>;
		using float_4x4 = short_matrix<float, 4, 4>;
		using float_1x3 = float_3;
		using float_2x3 = short_matrix<float, 2, 3>;
		using float_3x3 = short_matrix<float, 3, 3>;
		using float_4x3 = short_matrix<float, 4, 3>;
		using float_1x2 = float_2;
		using float_2x2 = short_matrix<float, 2, 2>;
		using float_3x2 = short_matrix<float, 3, 2>;
		using float_4x2 = short_matrix<float, 4, 2>;
		using float_1x1 = float;
		using float_2x1 = float_2;
		using float_3x1 = float_3;
		using float_4x1 = float_4;

		namespace direct3d
		{
			using matrix = float_4x4;
			using float1x4 = float_1x4;
			using float2x4 = float_2x4;
			using float3x4 = float_3x4;
			using float4x4 = float_4x4;
			using float1x3 = float_1x3;
			using float2x3 = float_2x3;
			using float3x3 = float_3x3;
			using float4x3 = float_4x3;
			using float1x2 = float_1x2;
			using float2x2 = float_2x2;
			using float3x2 = float_3x2;
			using float4x2 = float_4x2;
			using float1x1 = float_1x1;
			using float2x1 = float_2x1;
			using float3x1 = float_3x1;
			using float4x1 = float_4x1;
		}


#define __MAKE_BINARY_ASSIGN_OPERATOR(Opr) \
		template <typename T, int Rows, int Cols> \
		short_matrix<T,Rows,Cols>& operator Opr##= (short_matrix<T, Rows, Cols>& m0, short_matrix<T, Rows, Cols> m1) __GPU { \
			for (int r = 0; r < Rows; r++) \
				m0.r[r] Opr##= m1.r[r]; \
			return m0; \
		}

#define __MAKE_BINARY_ASSIGN_OPERATOR_TO_SCALAR(Opr) \
		template <typename T, int Rows, int Cols> \
		short_matrix<T,Rows,Cols>& operator Opr##= (short_matrix<T, Rows, Cols>& m0, T s) __GPU { \
			for (int r = 0; r < Rows; r++) \
				m0.r[r] Opr##= s; \
			return m0; \
		}

#define __MAKE_BINARY_OPERATOR(Opr) \
		template <typename T, int Rows, int Cols> \
		short_matrix<T,Rows,Cols> operator Opr (short_matrix<T, Rows, Cols> m0, short_matrix<T, Rows, Cols> m1) __GPU { \
			short_matrix<T, Rows, Cols> ret; \
			for (int r = 0; r < Rows; r++) \
				ret.r[r] = m0.r[r] Opr m1.r[r]; \
			return ret; \
		}

#define __MAKE_BINARY_OPERATOR_TO_SCALAR(Opr) \
		template <typename T, int Rows, int Cols> \
		short_matrix<T,Rows,Cols> operator Opr (short_matrix<T, Rows, Cols> m0, T s) __GPU { \
			short_matrix<T, Rows, Cols> ret; \
			for (int r = 0; r < Rows; r++) \
				ret.r[r] = m0.r[r] Opr s; \
			return ret; \
		}

		__MAKE_BINARY_OPERATOR(+)
		__MAKE_BINARY_OPERATOR(-)
		__MAKE_BINARY_OPERATOR(*)
		__MAKE_BINARY_OPERATOR(/)
		__MAKE_BINARY_OPERATOR_TO_SCALAR(*)
		__MAKE_BINARY_OPERATOR_TO_SCALAR(/)
		__MAKE_BINARY_ASSIGN_OPERATOR(+)
		__MAKE_BINARY_ASSIGN_OPERATOR(-)
		__MAKE_BINARY_ASSIGN_OPERATOR(*)
		__MAKE_BINARY_ASSIGN_OPERATOR(/)
		__MAKE_BINARY_ASSIGN_OPERATOR_TO_SCALAR(*)
		__MAKE_BINARY_ASSIGN_OPERATOR_TO_SCALAR(/)
	}
}