#pragma once
#include <Eigen\Dense>
#include <limits>
#include <tree.h>

namespace Causality
{
	class gaussian_process_lvm;
	class hierarchical_gaussian_process_lvm;
	// k(x,x') = a * exp(-0.5*c*dis(x,x')) + b*delta(x,x')
	// if theta = (a,b,r) , p(a,b,c) ~ a^-1*b^-1*c^-1 
	// L(Y_k |X, theta) = sum_k -ln p(Y_k | X, theta) - ln p(theta) = sum_k(0.5 * Y_k' * K^-1 * Y_k) + 0.5*D*ln|K| - ln p(theta)
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

		static constexpr double epsilon() {
			return std::numeric_limits<Scalar>::epsilon();
		};

		MatrixType X;
		MatrixType Y;
		RowVectorType wY; // Scale weights that applies to Y
		RowVectorType uX;
		RowVectorType uY;

	public:
		// Cache data
		Index N, D; // N = X.rows(), D = Y.cols();
		KernalMatrixType  Dx; // Dx(i,j) = -0.5 * |X(i,:) - X(j,:)|^2
		ParamType lparam;		// last theta
		KernalMatrixType K;			// The covarience matrix, positive semidefined symetric matrix NxX
		KernalMatrixType R;			// R = d(L_GP) / d(K) = (0.5 * D * K^-1) - (0.5 * K^-1 * Y * Y' *K^-1), grad(t) = tr(R * (dK/dt)) - d(ln p(theta))/dt
		MatrixType iKY;				// K^-1 * Y
		MatrixType iK;				// K^-1
		KernalMatrixType iKYYtiK;	// K^-1 * Y * Y' * K^-1 
		Eigen::LDLT<KernalMatrixType> ldltK;

		//mutable float detK;
		KernalMatrixType dKalpha;	// d(K)/d(alpha)
		KernalMatrixType dKgamma;	// d(K)/d(gemma)

	public:
		//template <class DerivedX, class DerivedY>
		gaussian_process_regression(const Eigen::MatrixXf& _X, const Eigen::MatrixXf& _Y);

		gaussian_process_regression();

		// initialize basic variables
		void initialize(const Eigen::MatrixXf& _X, const Eigen::MatrixXf& _Y);

		// initialize and automatic optimze parameter
		double fit_model(const Eigen::MatrixXf& _X, const Eigen::MatrixXf& _Y)
		{
			initialize(_X, _Y);
			return optimze_parameters();
		}

		inline double alpha() const
		{
			return lparam[0];
		}
		inline double beta() const
		{
			return lparam[1];
		}
		inline double gamma() const
		{
			return lparam[2];
		}

		inline const ParamType &get_parameters() const
		{
			return lparam;
		}

		Index sample_size() const { return N; }

		// Get the most-likly <y> from an gaussian noised observation of <x> : <z>, that P(x|z) ~ N(z,cov(X|Z))
		// return the likelihood of (y,x|z)
		double get_expectation_from_observation(_In_ const RowVectorType& z, _In_ const MatrixType &covXZ, _Out_ RowVectorType* y) const;

		double get_expectation_and_likelihood(_In_ const RowVectorType& x, _Out_ RowVectorType* y = nullptr) const;
		ColVectorType get_expectation_and_likelihood(_In_ const MatrixType& x, _Out_  MatrixType* y) const;

		void get_expectation(_In_ const RowVectorType& x, _Out_  RowVectorType* y) const;
		void get_expectation(_In_ const MatrixType& x, _Out_  MatrixType* y) const;

		// negitive log likilihood of P(y | theta,x)
		double get_likelihood_xy(const RowVectorType& x, const RowVectorType& y) const;
		RowVectorType get_likelihood_xy_derivative(const RowVectorType& x, const RowVectorType& y) const;

		double get_likelihood_x(const RowVectorType& x) const;
		RowVectorType get_ikelihood_x_derivative(const RowVectorType& x) const;

#pragma region Parameter Tuning Methods
		// negitive log likilihood of P(theta | X,Y)
		double likelihood(const ParamType &param);

		// gradiant of L
		ParamType likelihood_derivative(const ParamType &param);

		double optimze_parameters(const ParamType& initial_param);

		double optimze_parameters();

		inline void set_parameters(const ParamType &param)
		{
			update_kernal(param);
		}

#pragma endregion

	protected:
		void update_Dx();
		// aka. set parameter
		void update_kernal(const ParamType &param);
	};

	using gpr = gaussian_process_regression;

	// gaussian-process-latent-variable-model
	class gaussian_process_lvm : protected gaussian_process_regression
	{
	public:
		enum DynamicTypeEnum
		{
			NoDynamic = 0,
			OnePointPrior = 1,
			OnewayDynamic = 2,
			PeriodicDynamic = 3,
		};

		//using gpr::get_likelihood_xy;
		//using gpr::get_likelihood_x;
		using gpr::ParamSize;
		using gpr::MatrixType;
		using gpr::RowVectorType;
		using gpr::ColVectorType;
		using gpr::ParamType;

		using gpr::get_parameters;
		using gpr::alpha;
		using gpr::beta;
		using gpr::gamma;
		using gpr::sample_size;

		gaussian_process_lvm() { dyna_type = NoDynamic; }
		gaussian_process_lvm(const MatrixType& Y, Eigen::DenseIndex dX)
			:gaussian_process_lvm()
		{
			initialize(Y, dX);
		}


		// When sampleTimes == nullptr, we assume the sample are fixed interval sampled
		void set_dynamic(DynamicTypeEnum type, double timespan = 0, const ParamType* timeparam = nullptr, ColVectorType* sampleTimes = nullptr);
		void set_default(RowVectorType defautY, double weight);

		// aka. matrixX
		Index latent_dimension() const { return dimX; }
		const MatrixType& latent_coords() const;

		// L_IK(x,y|theta)
		double likelihood_xy(const RowVectorType& x, const RowVectorType& y) const;

		// grad(L_IK(x,y|theta))
		// return value layout = [dL/dx,dL/dy]
		RowVectorType likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const;

		// Allocate the storage for the model
		void initialize(const MatrixType& Y, Eigen::DenseIndex dX);

		// Load the model from external source,(e.g. files) to bypass the training
		double load_model(const MatrixType& X, const ParamType& param);

		// Train/learn the model with exited data Y
		double learn_model(const ParamType& param = ParamType(1.0, 1e3, 1.0), Scalar stop_delta = 1e-2, int max_iter = 100);

	public:
		template <typename DerivedX>
		void update_kernal(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType& param);
		// negitive log likilihood of P(X,theta | Y)
		template <typename DerivedX>
		double learning_likelihood(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType &param);

		// gradiant of L
		template <typename DerivedXOut, typename DerivedX>
		void learning_likelihood_derivative(_Out_ Eigen::MatrixBase<DerivedXOut>& dx, _Out_ ParamType& dparam, _In_ const Eigen::MatrixBase<DerivedX>& x, _In_ const ParamType &param);

		// optimize the latent coordinate X
		double optimize_x(const MatrixType& initalX);

		double optimze_parameters(const ParamType& initial_param);

		double optimze_parameters();
	protected:
		Index	   dimX;
		MatrixType dKx; // dK / dX_ij
		DynamicTypeEnum dyna_type;

		gpr			*parent;
		//ParamType  tparam;
		//Scalar	   tspan;
		//MatrixType Kt; // Kernal of time dynamic
		//MatrixType iKt; // K_t^-1
		//MatrixType iKtX; // K_t^-1 * X, gradiant L_GP to X under time constraint Kt
	};

	using gplvm = gaussian_process_lvm;

	class hierarchical_gaussian_process_lvm : stdx::tree_node<hierarchical_gaussian_process_lvm>, public gaussian_process_lvm
	{

	};

	// gaussian-process-shared-latent-variable-model
	class shared_gaussian_process_lvm
	{
	};
}