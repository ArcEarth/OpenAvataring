#pragma once
#include <Eigen\Dense>
#include <limits>
#include <tree.h>

namespace Causality
{
	class gaussian_process_lvm;
	class hierarchical_gaussian_process_lvm;
	// Gaussian Process with RBF kernal
	// Kernal : k(x,x') = a * exp(-0.5*c*dis(x,x')) + b*delta(x,x')
	// if theta = (a,b,r) , p(a,b,c) ~ a^-1*b^-1*c^-1 
	// For regressional prediction : 
	// L(Y_k |X, theta) = sum_k -ln p(Y_k | X, theta) - ln p(theta) = sum_k(0.5 * Y_k' * K^-1 * Y_k) + 0.5*dimY*ln|K| - ln p(theta)
	class gaussian_process_regression
	{
	public:
		friend class gaussian_process_lvm;
		friend class hierarchical_gaussian_process_lvm;

		using Scalar = double;
		using Index = Eigen::Index;
		static constexpr Index  ParamSize = 3;

		typedef Eigen::Vector3d ParamType;	// RBF Param type
		typedef Eigen::MatrixXd KernalMatrixType;
		typedef Eigen::MatrixXd MatrixType;
		typedef Eigen::RowVectorXd RowVectorType;
		typedef Eigen::VectorXd	ColVectorType;
		typedef Eigen::LDLT<KernalMatrixType> KernalLDLTType;

		// Process model data
		Index N, dimY, dimX; // N = X.rows(), dimY = Y.cols(), dimX = X.cols();
		MatrixType X;		 // input/latent variable samples
		MatrixType Y;		 // [zero meaned] output variable samples
		ParamType lparam;	 // RBF parameter
		KernalMatrixType K;  // The covarience matrix, positive semidefined symetric matrix NxN
		RowVectorType wY;	 // Scale weights that applies to Y
		RowVectorType uY; 	 // mean of Y_i,:
		RowVectorType uX;	 // mean of X, should be zero in most time

	public:
		KernalMatrixType Dx;		// Dx(i,j) = -0.5 * |X(i,:) - X(j,:)|^2
		KernalMatrixType R;			// R = d(L_GP) / d(K) = (0.5 * dimY * K^-1) - (0.5 * K^-1 * Y * Y' *K^-1), grad(t) = tr(R * (dK/dt)) - d(ln p(theta))/dt
		KernalMatrixType RcK;		// RcK = R .* K;
		MatrixType iKY;				// K^-1 * Y
		KernalMatrixType iK;		// K^-1
		KernalMatrixType iKYYtiK;	// K^-1 * Y * Y' * K^-1 
		KernalMatrixType dKalpha;	// d(K)/d(alpha)
		KernalMatrixType dKgamma;	// d(K)/d(gamma)
		KernalLDLTType	 ldltK; // det(K) == ldlt.dialog().product

	public:
		//template <class DerivedX, class DerivedY>
		gaussian_process_regression(const Eigen::MatrixXf& _X, const Eigen::MatrixXf& _Y);

		gaussian_process_regression();

		// allocate the storage
		void initialize(Index _N, Index _DimY, Index _DimX);
		// initialize basic variables
		void initialize(const Eigen::MatrixXf& _X, const Eigen::MatrixXf& _Y);

		// initialize and automatic optimze parameter
		double fit_model(const Eigen::MatrixXf& _X, const Eigen::MatrixXf& _Y)
		{
			initialize(_X, _Y);
			return optimze_parameters();
		}

		// the kernal output scale
		inline double alpha() const	{ return lparam[0];}
		// the white noise comes with trainning data
		inline double beta() const { return lparam[1]; }
		// the kernal internal variance
		inline double gamma() const { return lparam[2]; }
		inline const ParamType &get_parameters() const { return lparam; }

		Index sample_size() const { return N; }
		const MatrixType& samples() const { return Y; }
		Index output_dimension() const { return Y.size(); }
		Index latent_dimension() const { return dimX; }
		const MatrixType& latent_coords() const { return X; }

		// get the <expection> of <y> value under the condition of given laten coordinate x
		// return the negitive log uncertainty of this prediction
		// low value means good (close to trainning data)
		double get_ey_on_x(_In_ const RowVectorType& x, _Out_ RowVectorType* y = nullptr) const;
		// get the <expection(s)> of <Y> value under the condition of given laten coordinate(s) X
		// Batch version, overload for row-vector aggregation X
		// return the negitive log uncertainty of this prediction
		// low value means good (close to trainninget_likelihood_xyg data)
		ColVectorType get_ey_on_x(_In_ const MatrixType& x, _Out_  MatrixType* y) const;

		//void get_expectation(_In_ const RowVectorType& x, _Out_  RowVectorType* y) const;
		//void get_expectation(_In_ const MatrixType& x, _Out_  MatrixType* y) const;

		// negitive log likilihood of P(y,x|theta)
		double get_likelihood_xy(const RowVectorType& x, const RowVectorType& y) const;
		RowVectorType get_likelihood_xy_derivative(const RowVectorType& x, const RowVectorType& y) const;

		//double get_likelihood_x(const RowVectorType& x) const;
		//RowVectorType get_ikelihood_x_derivative(const RowVectorType& x) const;

		double optimze_parameters(const ParamType& initial_param);
		double optimze_parameters();
		inline void set_parameters(const ParamType &param)
		{
			update_kernal(param);
		}

		// Get the <most-likly-estimiton> of <y> from an gaussian noised observation of <x> : <z>, that P(x|z) ~ N(z,cov(X|Z))
		// return the likelihood of (y,x|z)
		double get_ey_on_obser_x(_In_ const RowVectorType& z, _In_ const MatrixType &covXZ, _Out_ RowVectorType* y) const;

	protected:
		void update_Dx();
		// aka. set parameter
		void update_kernal(const ParamType &param);

		template <typename DerivedX>
		void update_kernal(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType& param);

		// internal helper for caculate P(x,y|theta)
		void lp_xy_helper(const RowVectorType & x, RowVectorType & zx, ColVectorType &Kx, ColVectorType &iKkx, double &varX, RowVectorType &ey) const;
		// internal helper for caculate P(theta|x,y)
		double lp_param_on_xy();
		// internal helper for caculate P(theta|x,y)
		ParamType lp_param_on_xy_grad();

		// negitive log likilihood of P(theta | X,Y)
		double learning_likelihood_on_xy(const ParamType &param);
		// gradiant of negitive log likilihood of P(theta | X,Y)
		ParamType learning_likelihood_on_xy_derivative(const ParamType &param);
	};

	using gpr = gaussian_process_regression;

	// gaussian-process-latent-variable-model
	class gaussian_process_lvm : public gaussian_process_regression
	{
	private:
		using gpr::initialize;
		using gpr::fit_model;
	public:
		enum DynamicTypeEnum
		{
			NoDynamic = 0,
			OnePointPrior = 1,
			OnewayDynamic = 2,
			PeriodicDynamic = 3,
		};

		gaussian_process_lvm() { dyna_type = NoDynamic; parent = nullptr; }
		gaussian_process_lvm(const MatrixType& Y, Eigen::DenseIndex dX)
			:gaussian_process_lvm()
		{
			initialize(Y, dX);
		}


		// When sampleTimes == nullptr, we assume the sample are fixed interval sampled
		void set_dynamic(DynamicTypeEnum type, double timespan = 0, const ParamType* timeparam = nullptr, ColVectorType* sampleTimes = nullptr);
		void set_default(RowVectorType defautY, double weight);

		// L_IK(x,y|theta)
		double get_likelihood_xy(const RowVectorType& x, const RowVectorType& y) const;

		// grad(L_IK(x,y|theta))
		// return value layout = [dL/dx,dL/dy]
		RowVectorType get_likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const;

		// Allocate the storage for the model
		void initialize(const MatrixType& Y, Eigen::DenseIndex dX);

		// Load the model from external source,(e.g. files) to bypass the training
		double load_model(const MatrixType& X, const ParamType& param);

		// Train/learn the model with exited data Y
		double learn_model(const ParamType& param = ParamType(1.0, 1e-3, 1.0), Scalar stop_delta = 1e-2, int max_iter = 100);

	protected:
		template <typename DerivedX>
		void update_kernal(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType& param);
		// negitive log likilihood of P(X,theta | Y)
		template <typename DerivedX>
		double learning_likelihood(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType &param);

		// gradiant of L
		template <typename DerivedXOut, typename DerivedX>
		void learning_likelihood_derivative(_Out_ Eigen::MatrixBase<DerivedXOut>& dx, _Out_ ParamType& dparam, _In_ const Eigen::MatrixBase<DerivedX>& x, _In_ const ParamType &param);

	protected:
		DynamicTypeEnum dyna_type;
		gpr			*parent;
	};

	using gplvm = gaussian_process_lvm;

	class hierarchical_gaussian_process_lvm : stdx::tree_node<hierarchical_gaussian_process_lvm>, public gaussian_process_lvm
	{
	public:
		std::vector<int> m_childrenDim;
		std::vector<int> m_childrenDimStart;

	};

	// gaussian-process-shared-latent-variable-model
	class shared_gaussian_process_lvm
	{
	};

	template<typename DerivedX>
	inline void gaussian_process_regression::update_kernal(const Eigen::MatrixBase<DerivedX>& x, const ParamType & param)
	{
		X = x;
		uX = X.colwise().mean();
		update_Dx();
		update_kernal(param);
	}
}