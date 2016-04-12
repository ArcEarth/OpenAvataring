#pragma once
#include <Eigen\Core>
#include <algorithm>
#include <Geometrics\BezierClip.h>


namespace Eigen {

	enum SplineBoundryType
	{
		OpenLoop = 0,
		CloseLoop = 1, // Cyclic boundry
	};

	namespace Internal
	{
		constexpr DenseIndex prow(DenseIndex j)
		{
			return (j > 1) ? (2 * j + 1) : (j == 0 ? 0 : 3);
		}
	}

	template <class DerivedIn, class ArrayView, class DerivedOut>
	void selectCols(_In_ const DenseBase<DerivedIn>& In, _In_ const ArrayView& indices, _Out_ DenseBase<DerivedOut>* Out)
	{
		assert(Out->cols() == indices.size() && Out->rows() == In.rows());
		for (size_t i = 0; i < indices.size(); i++)
		{
			Out->col(i) = In.col(indices[i]);
		}
	};

	template <class DerivedIn, class ArrayView, class DerivedOut>
	//	typename Dummy = std::enable_if_t<std::is_integral<decltype(std::declval<ArrayView>()[0])>::value >
	void selectRows(_In_ const DenseBase<DerivedIn>& In, _In_ const ArrayView& indices, _Out_ DenseBase<DerivedOut>* Out)
	{
		assert(Out->rows() == indices.size() && Out->cols() == In.cols());
		for (size_t i = 0; i < indices.size(); i++)
		{
			Out->row(i) = In.row(indices[i]);
		}
	};

	template <class Derived>
	Eigen::Map<Matrix<typename Derived::Scalar, -1, -1, Derived::Options>>
		reshape(PlainObjectBase<Derived>& X, int rows = -1, int cols = -1)
	{
		if (rows == -1)
			rows = X.size() / cols;
		else if (cols == -1)
			cols = X.size() / rows;

		assert(X.size() == cols * rows);
		return Matrix<typename Derived::Scalar, -1, -1, Derived::Options>::Map(X.data(), rows, cols);
	}

	template <class Derived>
	Eigen::Map<const Matrix<typename Derived::Scalar, -1, -1, Derived::Options>>
		reshape(const PlainObjectBase<Derived>& X, int rows = -1, int cols = -1)
	{
		if (rows == -1)
			rows = X.size() / cols;
		else if (cols == -1)
			cols = X.size() / rows;

		assert(X.size() == cols * rows);
		return Matrix<typename Derived::Scalar, -1, -1, Derived::Options>::Map(X.data(), rows, cols);
	}

	// Apply Laplacian smooth in Row-wise sense (Each row is treated as a data point)
	template <class InputDerived>
	inline void laplacianSmooth(DenseBase<InputDerived> &X, double alpha = 0.8, unsigned IterationTimes = 2, SplineBoundryType boundryType = OpenLoop)
	{
		auto N = X.rows();
		using traits = internal::traits<InputDerived>;
		using Scalar = typename traits::Scalar;
		typedef Array<Scalar, traits::RowsAtCompileTime, traits::ColsAtCompileTime> MatrixType;
		typedef Map<MatrixType, 0, Stride<Dynamic, Dynamic>> MapType;
		MatrixType Cache(X.rows(), X.cols());

		MapType MapX(&X(0, 0), X.rows(), X.cols(), Stride<Dynamic, Dynamic>(X.outerStride(), X.innerStride()));
		MapType MapCache(Cache.data(), Cache.rows(), Cache.cols(), Stride<Dynamic, Dynamic>(Cache.outerStride(), Cache.innerStride()));
		MapType BUFF[2] = { MapX,MapCache };

		unsigned int src = 0, dst = 1;
		Scalar invAlpha = Scalar(0.5 * (1 - alpha));
		for (unsigned int k = 0; k < IterationTimes; k++)
		{
			if (boundryType == SplineBoundryType::OpenLoop)
			{
				BUFF[dst].row(0) = BUFF[src].row(0);
				BUFF[dst].row(N - 1) = BUFF[src].row(N - 1);
			}
			else // if (boundryType == SplineBoundryType::CloseLoop)
			{
				BUFF[dst].row(0) = alpha * BUFF[src].row(0) + invAlpha * (BUFF[src].row(N -1 ) + BUFF[src].row(1));
				BUFF[dst].row(N-1) = alpha * BUFF[src].row(0) + invAlpha * (BUFF[src].row(N - 2) + BUFF[src].row(0));
			}

			BUFF[dst].middleRows(1, N - 2) = alpha * BUFF[src].middleRows(1, N - 2) + invAlpha * ((BUFF[src].topRows(N - 2) + BUFF[src].bottomRows(N - 2)));
			dst = !dst;
			src = !src;
		}

		if (dst == 0)
		{
			X = Cache;
		}
	}

	// Cyclic boundary condition
	// Cublic-Bezier interpolation
	// Uniform sample assumption
	template <class OutputDerived, class InputDerived> inline
		void cublicBezierResample(DenseBase<OutputDerived> &Xr, const DenseBase<InputDerived> &X, DenseIndex nr, SplineBoundryType boundryType = OpenLoop)
	{
		using xtype = DenseBase<InputDerived>;
		using traits = internal::traits<InputDerived>;
		typedef typename traits::Scalar Scalar;
		using namespace std;
		typedef Stride<traits::OuterStrideAtCompileTime, traits::InnerStrideAtCompileTime == Dynamic ? Dynamic : traits::InnerStrideAtCompileTime * 2> StrideType;
		typedef Matrix<Scalar, Dynamic, traits::ColsAtCompileTime> StrideMatrixType;
		typedef Matrix<Scalar, 1, traits::ColsAtCompileTime> VectorType;
		typedef Matrix< typename internal::traits<InputDerived>::Scalar, Dynamic, traits::ColsAtCompileTime> ResampledMatrixType;
		typedef Geometrics::Bezier::BezierClipping<Matrix<Scalar, 1, traits::ColsAtCompileTime>, 3U> BezeirType;

		auto N = X.rows();
		auto K = X.cols();
		StrideMatrixType T(N, K); // Scaled Tagents
		T.topRows(N - 1) = X.middleRows(1, N - 1);
		if (boundryType == OpenLoop)
		{
			T.row(0) -= X.row(0);
			T.row(N - 1) = X.row(N - 1);
		}
		else
		{
			T.row(0) -= X.row(N - 1);
			T.row(N - 1) = X.row(0);
		}
		T.bottomRows(N - 1) -= X.middleRows(0, N - 1);
		if (boundryType == OpenLoop)
		{
			T.middleRows(1, N - 2) /= 6.0f;
			T.row(0) /= 3.0f;
			T.row(N - 1) /= 3.0f;
		}
		else
		{
			T /= 6.0f;
		}

		//cout << "T = \n" << T << endl;



		float ti;
		int Segs = N;
		if (boundryType == OpenLoop)
			ti = (float)(N - 1) / (float)(nr - 1);
		else
			ti = (float)(N) / (float)(nr);

		//ResampledMatrixType Xr(nr, K);
		Xr.resize(nr, K);
		BezeirType bezier;
		for (int i = 0, j = -1; i < nr; i++)
		{
			float r = fmodf(i * ti, 1.0f);
			auto k = (int)floorf(i * ti);

			if (k > j)
			{
				bezier[0] = X.row(k);
				bezier[3] = X.row((k + 1) % N);
				bezier[1] = bezier[0] + T.row(k);
				bezier[2] = bezier[3] - T.row((k + 1) % N);
				j = k;
			}
			Xr.row(i) = bezier(r);
		}


		//assert(!xtype::IsRowMajor);
		//assert(X.rows() >= 4 && X.rows() % 2 == 0);
		//if (X.rows() % 2 != 0)
		//{
		//	// Do something!!!
		//}

		//auto N = X.rows() / 2;
		//auto K = X.cols();

		//if (N == 2)
		//{ 
		//}

		//// Open-2xN path
		//Map<const StrideMatrixType, 0, StrideType> P(&X(3,0), N - 1, K, StrideType(X.outerStride(), X.innerStride() * 2));

		//// R has actually N-1 rows
		//Map<const StrideMatrixType, 0, StrideType> R(&X(4,0), N - 2, K, StrideType(X.outerStride(), X.innerStride() * 2));

		////cout << "P = \n" << P << endl;
		////cout << "R = \n" << R << endl;

		//StrideMatrixType T(N, K);

		//VectorType K0 = 27 * X.row(1) - 20 * X.row(0) - 7 * X.row(3);
		//VectorType K1 = 27 * X.row(2) - 20 * X.row(3) - 7 * X.row(0);

		//T.row(0) = K0 / 9.0f - K1 / 18.0f;
		//T.row(1) = K0 / 18.0f - K1 / 9.0f;

		//for (size_t i = 2; i < N; i++)
		//{
		//	T.row(i) = T.row(i - 1) - (4.0f / 3.0f) * (2 * R.row(i - 2) - P.row(i - 2) - P.row(i - 1));
		//}
		//
		////cout << "T = \n" << T << endl;

		//typedef Geometrics::Bezier::BezierClipping<Matrix<Scalar, 1, traits::ColsAtCompileTime>, 3U> BezeirType;
		//BezeirType bezier;

		//ResampledMatrixType Xr(nr,K);

		//float ti = (float)(N-0.5f) / (float)(nr - 1);

		//bezier[0] = X.row(0);
		//bezier[3] = X.row(3);
		//bezier[1] = bezier[0] + T.row(0);
		//bezier[2] = bezier[3] - T.row(1);

		//float r = 0;
		//for (int i = 0, j = 0; i < nr - 1; i++)
		//{
		//	if (i * ti >= 1.5f)
		//	{
		//		r = fmodf(i * ti - 1.5f, 1.0f);
		//		auto k = (int)floorf(i * ti - 0.5f);

		//		if (k > j)
		//		{
		//			bezier[0] = P.row(k - 1);
		//			bezier[3] = P.row(k);
		//			bezier[1] = bezier[0] + T.row(k);
		//			bezier[2] = bezier[3] - T.row(k + 1);
		//			j = k;
		//		}
		//		Xr.row(i) = bezier(r);
		//	}
		//	else
		//	{
		//		r = (i * ti) / 1.5f;
		//		Xr.row(i) = bezier(r);
		//	}
		//}
		//Xr.row(nr - 1) = X.row(X.rows() - 1);

		//return Xr;
	}

	namespace impl
	{
		template <class Derived>
		inline void compute_slack(
			const DenseIndex x,
			std::vector<typename Derived::Scalar>& slack,
			std::vector<DenseIndex>& slackx,
			const DenseBase<Derived> & cost,
			const std::vector<typename Derived::Scalar>& lx,
			const std::vector<typename Derived::Scalar>& ly
			)
		{
			auto n = ly.size();
			for (size_t y = 0; y < n; ++y)
			{
				if (lx[x] + ly[y] - cost(x, y) < slack[y])
				{
					slack[y] = lx[x] + ly[y] - cost(x, y);
					slackx[y] = x;
				}
			}
		}
	}

	template <class Derived> inline
		typename internal::traits<Derived>::Scalar
		matching_cost(const DenseBase<Derived> &cost, const std::vector<DenseIndex>& matching)
	{
		typedef typename internal::traits<Derived>::Scalar Scalar;
		Scalar sum(0);
		for (size_t i = 0; i < matching.size(); i++)
		{
			auto j = matching[i];
			if (j != -1)
				sum += cost(i, j);
		}
		return sum;
	}

	template <class Derived> inline
		std::vector<DenseIndex>
		max_weight_bipartite_matching(const DenseBase<Derived> &cost_)
	{
		typedef DenseBase<Derived> MatrixType;
		typedef typename internal::traits<Derived>::Scalar Scalar;
		typedef Scalar scalar_type;
		typedef DenseIndex index_type;
		using namespace impl;
		// Kuhn-Munkres Algorithm

		if (cost_.size() == 0)
			return std::vector<index_type>();

		size_t n = std::max(cost_.rows(), cost_.cols());
		Matrix<scalar_type, -1, -1> cost(n, n);
		cost.setZero();
		cost.block(0, 0, cost_.rows(), cost_.cols()) = cost_;

		std::vector<scalar_type> lx(n), ly(n);
		std::vector<index_type> xy;
		std::vector<index_type> yx;
		std::vector<char> S, T;
		std::vector<scalar_type> slack;
		std::vector<index_type> slackx;
		std::vector<index_type> aug_path;

		// Initially, nothing is matched. 
		xy.assign(n, -1);
		yx.assign(n, -1);
		/*
		We maintain the following invariant:
		Vertex x is matched to vertex xy[x] and
		vertex y is matched to vertex yx[y].

		A value of -1 means a vertex isn't matched to anything.  Moreover,
		x corresponds to rows of the cost matrix and y corresponds to the
		columns of the cost matrix.  So we are matching X to Y.
		*/

		// Create an initial feasible labeling.  Moreover, in the following
		// code we will always have: 
		//     for all valid x and y:  lx[x] + ly[y] >= cost(x,y)
		// Intialize flexable labels
		auto Lx = VectorXf::Map(lx.data(), n);
		Lx = cost.rowwise().maxCoeff();
		ly.resize(n);
		ly.assign(n, 0);

		// Now grow the match set by picking edges from the equality subgraph until
		// we have a complete matching.
		for (long match_size = 0; match_size < n; ++match_size)
		{
			std::deque<long> q;

			// Empty out the S and T sets
			S.assign(n, false);
			T.assign(n, false);

			// clear out old slack values
			slack.assign(n, std::numeric_limits<scalar_type>::max());
			slackx.resize(n);
			/*
			slack and slackx are maintained such that we always
			have the following (once they get initialized by compute_slack() below):
			- for all y:
			- let x == slackx[y]
			- slack[y] == lx[x] + ly[y] - cost(x,y)
			*/

			aug_path.assign(n, -1);

			for (long x = 0; x < n; ++x)
			{
				// If x is not matched to anything
				if (xy[x] == -1)
				{
					q.push_back(x);
					S[x] = true;

					compute_slack(x, slack, slackx, cost, lx, ly);
					break;
				}
			}


			long x_start = 0;
			long y_start = 0;

			// Find an augmenting path.  
			bool found_augmenting_path = false;
			while (!found_augmenting_path)
			{
				while (q.size() > 0 && !found_augmenting_path)
				{
					const long x = q.front();
					q.pop_front();
					for (long y = 0; y < n; ++y)
					{
						if (cost(x, y) == lx[x] + ly[y] && !T[y])
						{
							// if vertex y isn't matched with anything
							if (yx[y] == -1)
							{
								y_start = y;
								x_start = x;
								found_augmenting_path = true;
								break;
							}

							T[y] = true;
							q.push_back(yx[y]);

							aug_path[yx[y]] = x;
							S[yx[y]] = true;
							compute_slack(yx[y], slack, slackx, cost, lx, ly);
						}
					}
				}

				if (found_augmenting_path)
					break;


				// Since we didn't find an augmenting path we need to improve the 
				// feasible labeling stored in lx and ly.  We also need to keep the
				// slack updated accordingly.
				scalar_type delta = std::numeric_limits<scalar_type>::max();
				for (unsigned long i = 0; i < T.size(); ++i)
				{
					if (!T[i])
						delta = std::min(delta, slack[i]);
				}
				for (unsigned long i = 0; i < T.size(); ++i)
				{
					if (S[i])
						lx[i] -= delta;

					if (T[i])
						ly[i] += delta;
					else
						slack[i] -= delta;
				}



				q.clear();
				for (long y = 0; y < n; ++y)
				{
					if (!T[y] && slack[y] == 0)
					{
						// if vertex y isn't matched with anything
						if (yx[y] == -1)
						{
							x_start = slackx[y];
							y_start = y;
							found_augmenting_path = true;
							break;
						}
						else
						{
							T[y] = true;
							if (!S[yx[y]])
							{
								q.push_back(yx[y]);

								aug_path[yx[y]] = slackx[y];
								S[yx[y]] = true;
								compute_slack(yx[y], slack, slackx, cost, lx, ly);
							}
						}
					}
				}
			} // end while (!found_augmenting_path)

			  // Flip the edges aDenseIndex the augmenting path.  This means we will add one more
			  // item to our matching.
			for (long cx = x_start, cy = y_start, ty;
			cx != -1;
				cx = aug_path[cx], cy = ty)
			{
				ty = xy[cx];
				yx[cy] = cx;
				xy[cx] = cy;
			}

		}

		if (cost_.rows() < n)
			xy.resize(cost_.rows());
		else if (cost_.cols() < n)
			for (auto& xyi : xy)
				if (xyi >= cost_.cols())
					xyi = -1;

		return xy;

	}

}