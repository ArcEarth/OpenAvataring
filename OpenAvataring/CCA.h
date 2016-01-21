#pragma once
#include <Eigen\Dense>

namespace Eigen {
	// zero mean and Decompose X, thus (X + repmat(uX,n,1)) * E = [Q 0] * [R ; 0]
	template <class Derived>
	struct QrViewBase
	{
		typedef internal::traits<Derived> traits;
		typedef typename traits::Scalar Scalar;
		typedef typename traits::RealScalar RealScalar;
		typedef typename traits::Index Index;
		typedef typename traits::MatrixQReturnType MatrixQReturnType;
		typedef typename traits::MatrixRReturnType MatrixRReturnType;
		typedef typename traits::MeanReturnType MeanReturnType;
		typedef typename traits::ColsPermutationReturnType ColsPermutationReturnType;

		inline DenseIndex rows() const { return Derived::rows(); }
		inline DenseIndex cols() const { return Derived::cols(); }
		inline DenseIndex rank() const { return Derived::rank(); }
		inline MeanReturnType mean() const { return Derived::mean(); }
		inline MatrixQReturnType matrixQ() const { return Derived::matrixQ(); }
		inline MatrixRReturnType matrixR() const { return Derived::matrixR(); }
		inline ColsPermutationReturnType colsPermutation() const { return Derived::colsPermutation(); }
	};

	template< class _MatrixType>
	struct QrStore
	{
		typedef _MatrixType MatrixType;
		enum {
			RowsAtCompileTime = MatrixType::RowsAtCompileTime,
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			Options = MatrixType::Options,
			MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::Index Index;
		typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> MatrixQType;
		typedef Matrix<Scalar, Dynamic, Dynamic, Options, MaxColsAtCompileTime, MaxColsAtCompileTime> MatrixRType;
		typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime> PermutationType;
		typedef TriangularView<const MatrixRType, Upper> MatrixRTriangularViewType;
		typedef Matrix<Scalar, 1, ColsAtCompileTime, RowMajor, 1, MaxColsAtCompileTime> MeanVectorType;

		DenseIndex m_rows, m_cols, m_rank;	 // cols, rows, and rank
		MeanVectorType m_mean;			 // Rowwise Mean Vector of X, 1 x cols
		MatrixQType m_Q;		 // Self-Adjoint matrix Q , rows x rank
		MatrixRType m_R;		 // Upper triangluar matrix R , rank x rank
		PermutationType m_E; // column permutation matrix E , cols x cols

		// return if this Qr Stores the thin decomposition
		// thus cannot use QrView to partialy extract
		inline bool isThin() const { return m_Q.cols() < m_cols; }
		inline DenseIndex rows() const { return m_rows; }
		inline DenseIndex cols() const { return m_cols; }
		inline DenseIndex rank() const { return m_rank; }
		inline const MeanVectorType& mean() const { return m_mean; }
		inline const MatrixQType& matrixQ() const { return m_Q; }
		inline MatrixRTriangularViewType matrixR() const { return m_R.triangularView<Upper>(); }
		inline const MatrixRType& rawMatrixR() const { return m_R; }
		inline const PermutationType& colsPermutation() const { return m_E; }

		QrStore()
			: m_rank(0)
		{
		}

		template <class DerivedX>
		explicit QrStore(const DenseBase<DerivedX>& X, bool zeroMean = true, bool thinQR = true)
		{
			compute(X);
		}

		template <class DerivedX>
		void compute(const DenseBase<DerivedX>& X, bool zeroMean = true, bool thinQR = true)
		{
			if (X.size() == 0)
			{
				m_rank = 0;
				return;
			}

			m_cols = X.cols();
			m_rows = X.rows();

			MatrixType mX = X;
			if (zeroMean)
			{
				m_mean = X.colwise().mean().eval();
				mX.rowwise() -= m_mean; // dX x n
			}
			else
			{
				m_mean.setZero(X.cols());
			}

			// QR-decomposition
			auto qrX = mX.colPivHouseholderQr();

			m_rank = qrX.rank();
			m_E = qrX.colsPermutation();
			if (thinQR)
				m_R = qrX.matrixR().topLeftCorner(m_rank, m_rank).triangularView<Upper>();
			else
				m_R = qrX.matrixR();

			MatrixXf qX;
			qrX.matrixQ().evalTo(qX);
			if (thinQR)
				m_Q = qX.leftCols(m_rank);
		}

	};

	// light weight wrapper for view part of a QR decomposition
	template< class _MatrixType>
	struct QrView
	{
	public:
		typedef QrStore<_MatrixType> QrType;

		typedef _MatrixType MatrixType;
		enum {
			RowsAtCompileTime = MatrixType::RowsAtCompileTime,
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			Options = MatrixType::Options,
			MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::Index Index;
		typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> MatrixQType;
		typedef Matrix<Scalar, 1, ColsAtCompileTime, RowMajor, 1, MaxColsAtCompileTime> MeanVectorType;

		QrView(const QrType& qr, DenseIndex startRow = 0, DenseIndex rows = -1)
			: m_Qr(qr), m_sRow(startRow), m_rows(rows), m_rank(qr.rank())
		{
			if (m_rows == -1)
				m_rows = m_Qr.rows();
			assert(!m_Qr.isThin() || startRow == 0 && m_rows == m_Qr.rows() && "Thin Qr Store cannot be partialy extract");
		}

		inline DenseIndex rows() const { return m_rows; }
		inline DenseIndex cols() const { return m_Qr.cols(); }
		inline DenseIndex rank() const { return m_rank; }
		inline const MeanVectorType& mean() const { return m_Qr.mean(); }
		inline Eigen::Block<const MatrixQType> matrixQ() const { return m_Qr.matrixQ().block(m_sRow, m_sRow, m_rows, m_rank); }
		inline auto matrixR() const { return m_Qr.rawMatrixR().block(m_sRow, m_sRow, m_rank, m_rank).triangularView<Upper>(); }
		inline auto& colsPermutation() const { return m_Qr.colsPermutation(); }

	private:
		DenseIndex	  m_sRow, m_rows, m_rank;
		const QrType& m_Qr;
	};

	template<class _MatrixType>
	struct Pca
	{
	public:
		typedef _MatrixType MatrixType;
		enum {
			RowsAtCompileTime = MatrixType::RowsAtCompileTime,
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			Options = MatrixType::Options,
			MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::Index Index;

		typedef Matrix<Scalar, _MatrixType::ColsAtCompileTime, _MatrixType::ColsAtCompileTime> PrincipleComponentsType;
		typedef Matrix<Scalar, _MatrixType::RowsAtCompileTime, _MatrixType::ColsAtCompileTime> CordsMatrixType;
		typedef Matrix<Scalar, 1, _MatrixType::ColsAtCompileTime> RowVectorType;
	private:
		PrincipleComponentsType m_Comps;
		RowVectorType			m_Variences;
		RowVectorType			m_Mean;
		CordsMatrixType			m_Coords;
	public:
		Pca()
		{
		}

		explicit Pca(const MatrixType& X, bool computeCoords = true)
		{
			compute(X);
		}

		template <typename Derived>
		void compute(const MatrixBase<Derived>& X, bool computeCoords = true)
		{
			m_Mean = X.colwise().mean();
			auto Xz = (X.rowwise() - m_Mean).eval();

			auto svd = Xz.jacobiSvd(ComputeThinV);
			m_Comps = svd.matrixV();
			m_Variences = svd.singularValues();
			m_Variences = m_Variences.cwiseAbs2();

			if (computeCoords)
				m_Coords = Xz * m_Comps;
		}

		// compute pca from zero-mean data
		template <typename Derived>
		void computeCentered(const MatrixBase<Derived>& X, bool computeCoords = true)
		{
			m_Mean.setZero(X.cols());
			//JacobiSVD<Matrix<Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>> svd;
			//svd.compute(X,ComputeThinV);
			auto svd = X.jacobiSvd(ComputeThinV);

			m_Comps = svd.matrixV();
			m_Variences = svd.singularValues();
			m_Variences = m_Variences.cwiseAbs2();

			if (computeCoords)
				m_Coords = X * m_Comps;
		}

		// the demension after projection
		DenseIndex reducedRank(float cut_threshold_percentage) const
		{
			int rank;
			float cut_threshold = m_Variences[0] * cut_threshold_percentage;
			auto cols = m_Variences.size();
			for (rank = 0; rank < cols && m_Variences[rank] > cut_threshold; rank++);
			return rank;
		}

		void trim(DenseIndex nCols)
		{
			m_Comps.conservativeResize(m_Comps.rows(), nCols);
			m_Coords.conservativeResize(m_Coords.rows(), nCols);
			m_Variences.conservativeResize(nCols);
		}

		// column-wise principle components matrix
		auto components(DenseIndex nCols = -1) const {
			if (nCols < 0)
				nCols = m_Coords.cols();
			return m_Comps.leftCols(nCols);
		}
		// projected coordinates
		auto coordinates(DenseIndex nCols = -1) const
		{
			// you must spificy computeCoords = true in construction
			assert(m_Coords.size() > 0);

			if (nCols < 0)
				nCols = m_Coords.cols();
			return m_Coords.leftCols(nCols);
		}
		const auto& variences() const
		{
			return m_Variences;
		}

		const auto& mean() const { return m_Mean; }

		template <class Derived>
		void setMean(const DenseBase<Derived>& mean)
		{
			m_Mean = mean;
		}
	};

	// Canonical correlation analysis
	// template praram <_TScalar> : scalar used internal for this CCA
	// X : n x dX input vectors
	// Y : n x dY input vectors
	// n : number of observations
	// dX : dimension of feature X
	// dY : dimension of feature Y
	// return : correlation coefficients R, transform matrix A,B that maps X and Y to Latent space
	// thus argmax(A,B) corr(XA,YB)
	// it's best to make sure X Y are full-rank in the sense of colume rank
	// thus X' * A ~ Y' * B (X',Y' is zero meaned X,Y)
	template<class _TScalar>
	class Cca
	{
		typedef typename _TScalar Scalar;

		typedef Matrix<Scalar, -1, -1> MatrixType;
		typedef Matrix<Scalar, 1, -1> RowVectorType;
		typedef Matrix<Scalar, -1, 1> VectorType;

	private:
		bool m_initialized;
		MatrixType		A, B;	 // transform matrix to latent space
		VectorType		R;		 // Correlations
		RowVectorType	uX, uY;	 // mean vector of X and Y
		DenseIndex		rankX, rankY;	// rank of input X and Y
		DenseIndex		d, dX, dY;		// Dimension of latent space, X feature, Y feature

		ColPivHouseholderQR<MatrixType> qrX, qrY;
		JacobiSVD<MatrixType, 2> svdCovQXY;

		// for transform
		mutable JacobiSVD<MatrixType, 2> svdBt;
		mutable MatrixType				invB;
	public:
		Cca() : m_initialized(false) {}


		template <typename DerivedX, typename DerivedY>
		Cca(const DenseBase<DerivedX> &X, const DenseBase<DerivedY> &Y, bool computeAB = false)
		{
			compute(X, Y, computeAB);
		}

		template <typename DerivedX, typename DerivedY>
		Cca& compute(const DenseBase<DerivedX> &X, const DenseBase<DerivedY> &Y, bool computeAB = false);

		template<typename _MatrixTypeX, typename _MatrixTypeY>
		Cca& computeFromQr(const ColPivHouseholderQR<_MatrixTypeX>& qrX, const ColPivHouseholderQR<_MatrixTypeY>& qrY, bool computeAB = false);

		//template<typename _MatrixTypeX, typename _MatrixTypeY>
		//Cca& computeFromQr(const QrStore<_MatrixTypeX>& qrX, const QrStore<_MatrixTypeY>& qrY, bool computeAB = false);

		//template<typename DerivedX, typename DerivedY>
		//Cca& computeFromQr(const QrViewBase<DerivedX>& qrX, const QrViewBase<DerivedY>& qrY, bool computeAB = false);
		template<typename QrViewX, typename QrViewY>
		Cca& computeFromQr(const QrViewX& qrX, const QrViewY& qrY, bool computeAB = false, int rotY = 0);

		template<typename _MatrixTypeX, typename _MatrixTypeY>
		Cca& computeFromPca(const Pca<_MatrixTypeX>& pcaX, const Pca<_MatrixTypeY>& pcaY, bool computeAB = false);

		DenseIndex rank() const { return d; }
		const MatrixType& matrixA() const { return A; }
		const MatrixType& matrixB() const { return B; }
		const VectorType& correlaltions() const { return R; }

		// transform X into correlated Y
		//? this is not finished yet!!!
		MatrixXf transform(const MatrixXf& X) const
		{
			assert(X.cols() == dX); // dimension must agrees

			auto U = ((X.rowwise() - uX) * A).eval();

			MatrixXf Y;

			if (dY > d) // we need least square solution here
				Y = svdBt.solve(U.transpose()).transpose(); // Y' ~ B' \ U'
			else
				Y = U * invB; // Y = U / B

			Y.rowwise() += uY;
			return Y;
		}

		// transform Y into correlated X
		MatrixXf inv_transform(const MatrixXf& Y);

	};

#define DebugLog(mat) #mat << " = " << endl << mat << endl 

	template<typename _TScalar>
	template<typename _MatrixTypeX, typename _MatrixTypeY>
	inline Cca<_TScalar>& Cca<_TScalar>::computeFromQr(const ColPivHouseholderQR<_MatrixTypeX>& qrX, const ColPivHouseholderQR<_MatrixTypeY>& qrY, bool computeAB)
	{
		using namespace std;
		assert(qrX.rows() == qrX.rows());
		auto n = qrX.rows();
		// get ranks and latent space dimension
		rankX = qrX.rank();
		rankY = qrY.rank();
		dX = qrX.cols();
		dY = qrY.cols();


		d = min(rankX, rankY);

		// Get matrix Q,R and covQXY
		MatrixType qX, qY;
		qrX.matrixQ().evalTo(qX);
		qrY.matrixQ().evalTo(qY);
		//? This impl is not efficient!

		auto covQXY = (qX.leftCols(rankX).transpose() * qY.leftCols(rankY)).eval();

		//cout << "cov(Qx,Qy) = " << endl;
		//cout << covQXY << endl;

		// SVD
		unsigned int option = computeAB ? ComputeThinU | ComputeThinV : 0;
		auto svd = covQXY.jacobiSvd(option); // svdCovQXY

		R = svd.singularValues().topRows(d);

		//cout << DebugLog(R);

		if (computeAB)
		{
			auto rX = qrX.matrixR().topLeftCorner(rankX, rankX).triangularView<Upper>();
			auto rY = qrY.matrixR().topLeftCorner(rankY, rankY).triangularView<Upper>();

			A.setZero(dX, d);
			B.setZero(dY, d);

			// A = rX \ U * sqrt(n-1)
			// B = rY \ V * sqrt(n-1)
			A.topRows(rankX).noalias() = rX.solve(svd.matrixU().leftCols(d)) * sqrtf(n - 1.0f); // A : rankX x d
			B.topRows(rankY).noalias() = rY.solve(svd.matrixV().leftCols(d)) * sqrtf(n - 1.0f); // B : rankY x d
			// normalize A,B and reverse the permutation , thus U,V will have unit varience

			// Put coefficients back to their full size and their correct order
			A = qrX.colsPermutation().inverse() * A; // A : dX x d
			B = qrY.colsPermutation().inverse() * B; // B : dY x d

			//cout << DebugLog(A);
			//cout << DebugLog(B);

			// Compute Transform X -> Y
			if (d == dY)
				invB = B.inverse();
			else
				svdBt = JacobiSVD<MatrixType>(B.transpose());
		}

		m_initialized = true;
		return *this;
	}

	// it equals to the CCA of Up-Rotate(X,phi) -> Y
	template<typename _TScalar>
	template<typename QrViewX, typename QrViewY>
	inline Cca<_TScalar>& Cca<_TScalar>::computeFromQr(const QrViewX& qrX, const QrViewY& qrY, bool computeAB, int rotX = 0)
	{
		using namespace std;
		assert(qrX.rows() == qrX.rows());
		auto n = qrX.rows();
		// get ranks and latent space dimension
		rankX = qrX.rank();
		rankY = qrY.rank();
		dX = qrX.cols();
		dY = qrY.cols();

		uX = qrX.mean();
		uY = qrY.mean();

		d = min(rankX, rankY);
		if (d == 0)
			return *this;

		// Get matrix Q,R and covQXY
		MatrixType covQXY(rankX, rankY);

		if (rotX == 0)
			covQXY.noalias() = qrX.matrixQ().transpose() * qrY.matrixQ();
		else
			covQXY.noalias() = qrX.matrixQ().topRows(n - rotX).transpose() * qrY.matrixQ().bottomRows(n - rotX)
							 + qrX.matrixQ().bottomRows(rotX).transpose() * qrY.matrixQ().topRows(rotX);

		//std::cout << DebugLog(covQXY);

		// SVD
		unsigned int option = computeAB ? ComputeThinU | ComputeThinV : 0;
		auto svd = covQXY.jacobiSvd(option); // svdCovQXY

		R = svd.singularValues().topRows(d);

		//cout << DebugLog(R);

		if (computeAB)
		{
			auto rX = qrX.matrixR();
			auto rY = qrY.matrixR();

			A.setZero(dX, d);
			B.setZero(dY, d);

			// A = rX \ U * sqrt(n-1)
			// B = rY \ V * sqrt(n-1)
			A.topRows(rankX).noalias() = rX.solve(svd.matrixU().leftCols(d)) * sqrtf(n - 1.0f); // A : rankX x d
			B.topRows(rankY).noalias() = rY.solve(svd.matrixV().leftCols(d)) * sqrtf(n - 1.0f); // B : rankY x d
			//std::cout << DebugLog(svd.matrixU());
			//std::cout << DebugLog(svd.matrixV());

			//std::cout << DebugLog(A);
			//std::cout << DebugLog(B);

			// Put coefficients back to their full size and their correct order
			A = qrX.colsPermutation() * A; // A : dX x d
			B = qrY.colsPermutation() * B; // B : dY x d

			//cout << DebugLog(A);
			//cout << DebugLog(B);

			// Compute Transform X -> Y
			if (d == dY)
				invB = B.inverse();
			else
				svdBt = JacobiSVD<MatrixXf>(B.transpose());
		}

		m_initialized = true;
		return *this;
	}



	template<class _TScalar>
	template <typename DerivedX, typename DerivedY>
	inline Cca<_TScalar>& Cca<_TScalar>::compute(const DenseBase<DerivedX> &X, const DenseBase<DerivedY> &Y, bool computeAB)
	{
		// Algorithm is explianed here : ( Qr + SVD version)
		// http://www.nr.com/whp/notes/CanonCorrBySVD.pdf
		assert(X.rows() == Y.rows());
		auto n = X.rows();
		dX = X.cols();
		dY = Y.cols();

		// zero mean X and Y
		uX = X.colwise().mean().eval();
		uY = Y.colwise().mean().eval();
		MatrixType mX = X.rowwise() - uX; // dX x n
		MatrixType mY = Y.rowwise() - uY; // dY x n

		// QR-decomposition
		auto qrX = mX.colPivHouseholderQr();
		auto qrY = mY.colPivHouseholderQr();

		return computeFromQr(qrX, qrY, computeAB);
	}

	namespace internal
	{
		//template <class _MatrixType>
		//struct traits<QrStore<_MatrixType>>
		//{
		//	typedef _MatrixType MatrixType;
		//	enum {
		//		RowsAtCompileTime = MatrixType::RowsAtCompileTime,
		//		ColsAtCompileTime = MatrixType::ColsAtCompileTime,
		//		Options = MatrixType::Options,
		//		MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
		//		MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		//	};

		//	typedef typename MatrixType::Scalar Scalar;
		//	typedef typename MatrixType::RealScalar RealScalar;
		//	typedef typename MatrixType::Index Index;
		//	typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> MatrixQType;
		//	typedef Matrix<Scalar, Dynamic, Dynamic, Options, MaxColsAtCompileTime, MaxColsAtCompileTime> MatrixRType;
		//	typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime> PermutationType;
		//	typedef Matrix<Scalar, 1, ColsAtCompileTime, RowMajor, 1, MaxColsAtCompileTime> MeanVectorType;

		//	typedef const MatrixQType& MatrixQReturnType;
		//	typedef TriangularView<const MatrixRType, Upper> MatrixRReturnType;
		//	typedef const MeanVectorType& MeanReturnType;
		//	typedef const PermutationType& ColsPermutationReturnType;
		//};
		//template <class _MatrixType>
		//struct traits<MeanThickQr<_MatrixType>>
		//{
		//	typedef _MatrixType MatrixType;
		//	enum {
		//		RowsAtCompileTime = MatrixType::RowsAtCompileTime,
		//		ColsAtCompileTime = MatrixType::ColsAtCompileTime,
		//		Options = MatrixType::Options,
		//		MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
		//		MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		//	};

		//	typedef typename MatrixType::Scalar Scalar;
		//	typedef typename MatrixType::RealScalar RealScalar;
		//	typedef typename MatrixType::Index Index;
		//	typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> MatrixQType;
		//	typedef Matrix<Scalar, 1, ColsAtCompileTime, RowMajor, 1, MaxColsAtCompileTime> MeanVectorType;
		//	typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime> PermutationType;

		//	typedef const MatrixQType& MatrixQReturnType;
		//	typedef Eigen::TriangularView<MatrixQType, Upper> MatrixRReturnType;
		//	typedef const MeanVectorType& MeanReturnType;
		//	typedef const PermutationType& ColsPermutationReturnType;

		//};
		//template <class _MatrixType>
		//struct traits<QrView<_MatrixType>>
		//{
		//	typedef _MatrixType MatrixType;
		//	enum {
		//		RowsAtCompileTime = MatrixType::RowsAtCompileTime,
		//		ColsAtCompileTime = MatrixType::ColsAtCompileTime,
		//		Options = MatrixType::Options,
		//		MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
		//		MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		//	};

		//	typedef typename MatrixType::Scalar Scalar;
		//	typedef typename MatrixType::RealScalar RealScalar;
		//	typedef typename MatrixType::Index Index;
		//	typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> MatrixQType;
		//	typedef Matrix<Scalar, 1, ColsAtCompileTime, RowMajor, 1, MaxColsAtCompileTime> MeanVectorType;
		//	typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime> PermutationType;

		//	typedef Eigen::Block<const MatrixQType> MatrixQReturnType;
		//	typedef Eigen::TriangularView<const Eigen::Block<const MatrixQType>, Upper> MatrixRReturnType;
		//	typedef const MeanVectorType& MeanReturnType;
		//	typedef const PermutationType& ColsPermutationReturnType;
		//};
	}
}