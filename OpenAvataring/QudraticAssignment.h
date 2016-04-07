#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

#include <span.h>
// spamming std namespace
namespace std
{
	template <typename Iterator>
	inline bool next_combination(Iterator first,
		Iterator k,
		Iterator last);

	template <typename Iterator>
	inline bool next_combination(const Iterator first, Iterator k, const Iterator last)
	{
		/* Credits: Thomas Draper */
		// http://stackoverflow.com/a/5097100/8747
		if ((first == last) || (first == k) || (last == k))
			return false;
		Iterator itr1 = first;
		Iterator itr2 = last;
		++itr1;
		if (last == itr1)
			return false;
		itr1 = last;
		--itr1;
		itr1 = k;
		--itr2;
		while (first != itr1)
		{
			if (*--itr1 < *itr2)
			{
				Iterator j = k;
				while (!(*itr1 < *j)) ++j;
				std::iter_swap(itr1, j);
				++itr1;
				++j;
				itr2 = k;
				std::rotate(itr1, j, last);
				while (last != j)
				{
					++j;
					++itr2;
				}
				std::rotate(k, itr2, last);
				return true;
			}
		}
		std::rotate(first, k, last);
		return false;
	}
}

namespace Eigen
{
	// C(i,j,ass(i),ass(j)) must exist
	template <class MatrixType, class QuadraticFuncType, typename IndexType>
	float quadratic_assignment_cost(const MatrixType& A, const QuadraticFuncType &C, const IndexType *ass, bool transposed)
	{
		using namespace std;

		IndexType n = min(A.rows(),A.cols());
		float score = .0f;

		if (!transposed)
		{
			for (IndexType i = 0; i < n; i++)
			{
				score += A(i, ass[i]);
				for (IndexType j = i + 1; j < n; j++)
				{
					float c = C(i, j, ass[i], ass[j]);
					score += c;
				}
			}
		}
		else
		{
			for (IndexType i = 0; i < n; i++)
			{
				score += A(ass[i], i);
				for (IndexType j = i + 1; j < n; j++)
				{
					float c = C(ass[i], ass[j], i, j);
					score += c;
				}
			}
		}

		return score;
	}

	template <class MatrixType, typename IndexType>
	float assignment_cost(const MatrixType& A, const IndexType *ass, bool transposed)
	{
		using namespace std;

		IndexType n = min(A.rows(), A.cols());
		float score = .0f;

		if (!transposed)
		{
			for (IndexType i = 0; i < n; i++)
			{
				score += A(i, ass[i]);
			}
		}
		else
		{
			for (IndexType i = 0; i < n; i++)
			{
				score += A(ass[i], i);
			}
		}

		return score;
	}

	// C(i,j,ass(i),ass(j)) must exist
	// Brute-force solve QAP
	template <class MatrixType, class QuadraticFuncType, typename IndexType>
	float max_quadratic_assignment(const MatrixType& A, const QuadraticFuncType &C, _Out_ gsl::span<IndexType> assignment)
	{
		using namespace std;

		constexpr IndexType null_assign = static_cast<IndexType>(-1);

		// in this case, we do a transposed question
		bool transposed = A.rows() > A.cols();

		size_t nx = A.rows(), ny = A.cols();
		if (transposed)
			swap(nx, ny);

		vector<IndexType> s(ny);
		iota(s.begin(), s.end(), 0);

		vector<IndexType>  optAss((size_t)nx, null_assign);
		float optScore = std::numeric_limits<float>::min();

		do {
			do {
				float score = quadratic_assignment_cost(A, C, s.data(), transposed);
				//if (std::isnan(score))
				//{
				//	_CrtDbgBreak();
				//	cout << "Bug lurks here!" << endl;
				//}

				if (score > optScore)
				{
					optAss.assign(s.begin(), s.begin() + nx);
					optScore = score;
				}
#ifdef _DEBUG_QAP
				cout << "Assignment ";
				if (transposed)
					cout << "(B->A) : ["; else cout << "(A->B) : [";

				for (auto& i : std::make_range(s.begin(),s.begin() + nx))
					cout << i << ',';
				float asScore = assignment_cost(A, s.data(), transposed);
				cout << "\b] = (" << asScore << " + " << score - asScore << ')' << endl;
#endif
			} while (std::next_permutation(s.begin(), s.begin() + nx));
		} while (next_combination(s.begin(), s.begin() + nx, s.end()));

		if (!transposed)
			std::copy_n(optAss.begin(), optAss.size(), assignment.begin());
			//assignment = optAss;
		else
		{
			//assignment.resize(A.rows());
			std::fill(assignment.begin(), assignment.end(), null_assign);

			//cout << "optAss = ";
			for (int i = 0; i < optAss.size(); i++)
			{
				//cout << optAss[i] << ' ';
				if (optAss[i] >= 0 && optAss[i] < assignment.size())
					assignment[optAss[i]] = i;
			}
			//cout << endl;
		}

		return optScore;
	}

}