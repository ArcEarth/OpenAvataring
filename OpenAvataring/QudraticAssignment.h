#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

#include <Eigen\Core>

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
	template <class QuadraticFuncType>
	float quadratic_assignment_cost(const Eigen::MatrixXf& A, const QuadraticFuncType &C, _In_reads_(A.rows()) Eigen::DenseIndex* ass, bool transposed)
	{
		using namespace std;
		using namespace Eigen;

		int n = min(A.rows(),A.cols());
		float score = 0;

		for (int i = 0; i < n; i++)
		{
			score += transposed ? A(ass[i],i) : A(i, ass[i]);
			for (int j = i + 1; j < n; j++)
			{
				score += transposed ? C(ass[i], ass[j], i, j) : C(i, j, ass[i], ass[j]);
			}
		}

		return score;
	}

	// C(i,j,ass(i),ass(j)) must exist
	// Brute-force solve QAP
	template <class QuadraticFuncType>
	float max_quadratic_assignment(const Eigen::MatrixXf& A, const QuadraticFuncType &C, _Out_ std::vector<Eigen::DenseIndex>& assignment)
	{
		using namespace std;
		using namespace Eigen;

		// in this case, we do a transposed question
		bool transposed = A.rows() > A.cols();

		auto nx = A.rows(), ny = A.cols();
		if (transposed)
			swap(nx, ny);

		vector<DenseIndex> s(std::max(nx, ny));
		iota(s.begin(), s.end(), 0);

		vector<DenseIndex>  optAss(nx);
		std::fill(optAss.begin(), optAss.end(), -1);
		float optScore = std::numeric_limits<float>::min();

		do {
			do {
				float score = quadratic_assignment_cost(A, C, s.data(), transposed);
				if (score > optScore)
				{
					optAss.assign(s.begin(), s.begin() + nx);
					optScore = score;
				}
#ifdef _DEBUG
				for (auto& i : s)
				{
					cout << i << ' ';
				}
				cout << ':' << score << endl;
#endif
			} while (std::next_permutation(s.begin(), s.begin() + nx));
		} while (next_combination(s.begin(), s.begin() + nx, s.end()));

		if (!transposed)
			assignment = optAss;
		else
		{
			assignment.resize(A.rows());
			fill(assignment.begin(), assignment.end(), -1);

			for (int i = 0; i < optAss.size(); i++)
				assignment[optAss[i]] = i;
		}

		return optScore;
	}

}