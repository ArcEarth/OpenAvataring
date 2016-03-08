#include "pch.h"
#include "GaussianProcess.h"
// For Pca / Cca
#include "CCA.h"
#include <fstream>
// For Optimization like Conjucate Gradiant
#include <dlib\optimization\optimization.h>

//#include <unsupported/Eigen/NonLinearOptimization>
//#include <unsupported/Eigen/NumericalDiff>

using namespace Causality;

double g_paramMin = 1e-7;
double g_paramMax = 1e7;
double g_paramWeight = 1;

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

	int m_inputs, m_values;

	Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }

};


typedef dlib::matrix<double, 0, 1> dlib_vector;

double dlib_likelihood(gaussian_process_regression* pThis, const dlib_vector& m);

// This is a helper function used while optimizing the rosen() function.  
dlib_vector dlib_likelihood_derv(gaussian_process_regression* pThis, const dlib_vector& m);

double dlib_likelihood(gaussian_process_regression* pThis, const dlib_vector& m)
{
	gaussian_process_regression::ParamType param(m(0), m(1), m(2));
	return pThis->likelihood(param);
}

dlib_vector dlib_likelihood_derv(gaussian_process_regression* pThis, const dlib_vector& m)
{
	gaussian_process_regression::ParamType param(m(0), m(1), m(2));
	auto grad = pThis->likelihood_derivative(param);

	dlib_vector result = m;
	result(0) = grad(0);
	result(1) = grad(1);
	result(2) = grad(2);
	return result;
}

gaussian_process_regression::gaussian_process_regression(const Eigen::MatrixXf & _X, const Eigen::MatrixXf & _Y)
//: DenseFunctor<float, 3, 1>()
{
	initialize(_X, _Y);
}

gaussian_process_regression::gaussian_process_regression()
{
	dimX = 0; N = 0; D = 0;
}

void gaussian_process_regression::initialize(const Eigen::MatrixXf & _X, const Eigen::MatrixXf & _Y)
{
	assert(_X.rows() == _Y.rows() && _X.rows() > 0 && "Observations count agree");
	assert(!_X.hasNaN());

	dimX = _X.cols();
	N = _X.rows();
	D = _Y.cols();

	X = _X.cast<double>();
	Y = _Y.cast<double>();
	uX = X.colwise().mean();
	uY = Y.colwise().mean();
	X = X - uX.replicate(N, 1);
	Y = Y - uY.replicate(N, 1);

	Dx.resize(N, N);

	K.resize(N, N);
	R.resize(N, N);
	iKY.resize(N, Y.cols());
	iKYYtiK.resize(N, N);

	update_Dx();
}

void gaussian_process_regression::update_Dx()
{
	for (int i = 0; i < N; i++)
	{
		Dx.row(i) = (-0.5f) * (X.row(i).replicate(N, 1) - X).rowwise().squaredNorm().transpose();
	}
}

void gaussian_process_regression::update_kernal(const ParamType & param)
{
	//if ((lparam - param).cwiseAbs().sum() < 1e-8)
	//	return;

	lparam = param;

	dKalpha.array() = (Dx * gamma()).array().exp();
	K = (alpha() * dKalpha);
	K.diagonal().array() += beta();

	ldltK.compute(K);
	assert(ldltK.info() == Eigen::Success);

	iK = ldltK.solve(Eigen::MatrixXd::Identity(N, N));
	iKY.noalias() = iK * Y;//ldltK.solve(Y);
}

double gaussian_process_regression::get_expectation_from_observation(const RowVectorType & z, _In_ const MatrixType &covXZ, RowVectorType * y) const
{
	// Center Z with the mean
	assert(z.size() == covXZ.rows() && covXZ.rows() == covXZ.cols() && "CovXZ should be positive semi-defined");
	RowVectorType cZ = z - uX;
	auto ldltCov = covXZ.ldlt();
	auto invCov = ldltCov.solve(MatrixType::Identity(z.size(), z.size())).eval();
	auto detCov = ldltCov.vectorD().prod();

	struct ObsvExpectionFunctor : public Functor<double>
	{
		const RowVectorType& cZ;
		const Eigen::MatrixXd& invCov;
		const gaussian_process_regression& _this;

		ObsvExpectionFunctor(const gaussian_process_regression& gpr, const RowVectorType& _cZ, const Eigen::MatrixXd& _invCov)
			: _this(gpr), cZ(_cZ), invCov(_invCov), Functor<double>(_cZ.size(), 1)
		{
		}

		int operator()(const ColVectorType& _x, ColVectorType& fvec)
		{
			RowVectorType x = _x.transpose();
			//auto x = RowVectorType::Map(_x.begin(), _x.end() - _x.begin());

			ColVectorType Kx = (x.replicate(_this.N, 1) - _this.X).cwiseAbs2().rowwise().sum();
			Kx.array() = ((-0.5*_this.gamma()) * Kx.array()).exp() * _this.alpha();

			ColVectorType iKkx = _this.iK * Kx;//ldltK.solve(Kx);
			double cov = Kx.transpose() * iKkx;
			double varX = _this.alpha() + _this.beta() - cov;
			varX = std::max(varX, 1e-5);
			assert(varX > 0);
			RowVectorType dxCz = x - cZ;
			double value = 0.5 * _this.D * log(varX) + 0.5 * dxCz * invCov * dxCz.transpose();

			fvec(0) = value;
			//return value;
			return 0;
		};

		int df(const ColVectorType& _x, Eigen::MatrixXd& fjac)
		{
			//auto x = RowVectorType::Map(_x.begin(), _x.end() - _x.begin());
			RowVectorType x = _x.transpose();

			// N x d
			MatrixType dx = x.replicate(_this.N, 1) - _this.X;
			// N x 1
			ColVectorType Kx = dx.cwiseAbs2().rowwise().sum();
			Kx = ((-0.5*_this.gamma()) * Kx.array()).exp() * _this.alpha();
			// N x d
			MatrixType dKxx = dx.array() * Kx.replicate(1, dx.cols()).array();

			ColVectorType iKkx = _this.iK * Kx;//ldltK.solve(Kx);
			double cov = Kx.transpose() * iKkx;
			double varX = _this.alpha() + _this.beta() - cov;

			RowVectorType derv = iKkx.transpose() * dKxx;
			derv = -_this.D / varX * derv + (x - cZ) * invCov; // / varZ;

			fjac = derv;
			return 0;
			//return derv.transpose();
			//dlib_vector _dx(derv.cols());
			//for (int i = 0; i < derv.cols(); i++)
			//{
			//	_dx(i) = derv(i);
			//}

			//return _dx;
		};
	} functor(*this, cZ, invCov);

	auto f = [this, &cZ, &invCov](const dlib_vector& _x) -> double
	{
		auto x = RowVectorType::Map(_x.begin(), _x.end() - _x.begin());

		ColVectorType Kx = (x.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
		Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

		ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
		double cov = Kx.transpose() * iKkx;
		double varX = alpha() + beta() - cov;
		varX = std::max(varX, 1e-5);
		assert(varX > 0);
		RowVectorType dxCz = x - cZ;
		double value = 0.5 * D * log(varX) + 0.5 * dxCz * invCov * dxCz.transpose();

		return value;
	};

	auto df = [this, &cZ, &invCov](const dlib_vector& _x) -> dlib_vector
	{
		auto x = RowVectorType::Map(_x.begin(), _x.end() - _x.begin());

		// N x d
		MatrixType dx = x.replicate(N, 1) - X;
		// N x 1
		ColVectorType Kx = dx.cwiseAbs2().rowwise().sum();
		Kx = ((-0.5*gamma()) * Kx.array()).exp() * alpha();
		// N x d
		MatrixType dKxx = dx.array() * Kx.replicate(1, dx.cols()).array();

		ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
		double cov = Kx.transpose() * iKkx;
		double varX = alpha() + beta() - cov;

		RowVectorType derv = iKkx.transpose() * dKxx;
		derv = -D / varX * derv + (x - cZ) * invCov; // / varZ;

		dlib_vector _dx(derv.cols());
		for (int i = 0; i < derv.cols(); i++)
		{
			_dx(i) = derv(i);
		}

		return _dx;
	};

	// initalize _x to (x - uX)
	dlib_vector _x(X.cols());
	for (int i = 0; i < X.cols(); i++)
	{
		_x(i) = X(i) - uX(i);
	}
	//ColVectorType _x = X - uX.transpose();

	//// Test anaylatic derivative
	//auto numberic_diff = dlib::derivative(f)(_x);
	//auto anaylatic_diff = df(_x);

	//std::cout << "numberic derv = " << dlib::trans(numberic_diff) << "anaylatic derv = " << dlib::trans(anaylatic_diff) << std::endl;
	//std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
	//	<< dlib::length(numberic_diff - anaylatic_diff) << std::endl;

	//Eigen::LevenbergMarquardt<ObsvExpectionFunctor,double> lm(functor);
	//lm.minimize(_x);
	//double likelihood = lm.fvec(0);
	double likelihood = dlib::find_min(
		dlib::cg_search_strategy(),
		dlib::objective_delta_stop_strategy(1e-3),
		f, df,//dlib::derivative(f)
		_x,
		std::numeric_limits<double>::min());

	//RowVectorType eX = _x.transpose() + uX;
	RowVectorType eX = RowVectorType::Map(_x.begin(), _x.end() - _x.begin()) + uX;

	//std::cout << "Optimzation over L(X|Z=" << z << "), yield E(X) = " << eX << ", -Log(L) = " << likelihood << std::endl;

	get_expectation(eX, y);

	likelihood = exp(-likelihood);

	return likelihood;
}

double gaussian_process_regression::get_expectation_and_likelihood(const RowVectorType & x, RowVectorType * y) const
{
	assert(x.cols() == X.cols());

	RowVectorType dx = x.cast<double>() - uX;
	ColVectorType Kx = (dx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	double varX = alpha() + beta() - cov;

	//assert(varX > 0);

	if (y != nullptr)
		*y = uY + iKkx.transpose() * Y;

	return 0.5 * D * log(abs(varX));
}

gaussian_process_regression::ColVectorType gaussian_process_regression::get_expectation_and_likelihood(const MatrixType & x, MatrixType * y) const
{
	assert(x.cols() == X.cols());

	double varX = alpha() + beta();

	MatrixType dx = x - uX.replicate(x.rows(), 1);
	MatrixType Kx(N, x.rows());
	for (size_t i = 0; i < x.rows(); i++)
	{
		Kx.col(i) = (-0.5*gamma()) * (dx.row(i).replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	}
	Kx = Kx.array().exp() * alpha(); // N x m, m = x.rows()

	MatrixType iKkx = iK * Kx;//ldltK.solve(Kx); // N x m

	if (y != nullptr)
	{
		*y = uY.replicate(x.rows(), 1) + iKkx.transpose() * Y;
	}

	ColVectorType cov = (Kx.array() * iKkx.array()).colwise().sum().transpose();
	cov = (0.5 * D) * (varX - cov.array()).abs().log();


	return cov;


}

void gaussian_process_regression::get_expectation(const RowVectorType & x, RowVectorType * y) const
{
	assert(x.cols() == X.cols());

	RowVectorType dx = x.cast<double>() - uX;
	ColVectorType Kx = (dx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	if (y != nullptr)
	{
		ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
		*y = uY + iKkx.transpose() * Y;
		//*y = uY + (Y.transpose() * ldltK.solve(Kx)).transpose();
	}
}

void gaussian_process_regression::get_expectation(const MatrixType & x, MatrixType * y) const
{
	assert(x.cols() == X.cols());
	MatrixType dx = x - uX.replicate(x.rows(), 1);
	MatrixType Kx(x.rows(), N);
	for (size_t i = 0; i < x.rows(); i++)
	{
		Kx.row(i) = (-0.5*gamma()) * (dx.row(i).replicate(N, 1) - X).cwiseAbs2().rowwise().sum().transpose();
	}
	Kx.array() = (Kx.array()).exp() * alpha(); // m x N, m = x.rows()

	if (y != nullptr)
	{
		MatrixType iKkx = (iK * Kx.transpose()).transpose();//ldltK.solve(Kx.transpose()).transpose(); // m X N
		*y = uY.replicate(x.rows(), 1) + iKkx * Y;
	}
}

// negitive log likilihood of P(y | theta,x)
double gaussian_process_regression::get_likelihood_xy(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType ey(y.size());
	double varX = alpha() + beta();

	RowVectorType dx = x.cast<double>() - uX;
	ColVectorType Kx = (dx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	varX = std::max(1e-5, varX - cov);

	assert(varX > 0);

	ey = uY + iKkx.transpose() * Y;

	double difY = (y - ey).cwiseAbs2().sum();

	double lxy = 0.5*(difY / varX + D * log(varX));

	return lxy;
}

gaussian_process_regression::RowVectorType gaussian_process_regression::get_likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType ey(y.size());

	RowVectorType zx = x.cast<double>() - uX;
	auto dKx = (zx.replicate(N, 1) - X).eval();

	ColVectorType Kx = dKx.cwiseAbs2().rowwise().sum();
	Kx = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	dKx = Kx.asDiagonal() * dKx;

	ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	double varX = alpha() + beta() - cov; // sigma^2(x)

	ey = uY + iKkx.transpose() * Y;

	RowVectorType derv(x.size() + y.size());
	auto difY = (y - ey).eval();

	auto dy = derv.segment(x.size(), y.size());
	dy = difY / varX;

	auto dx = derv.segment(0, x.size());

	auto dfx = Y.transpose() * iK * dKx;

	auto dvarx = 2 * Kx * iK * dKx;

	dx = difY * dfx / varX - (difY.squaredNorm() / varX + D) / (2 * varX) * dvarx;

	return derv;
}


double gaussian_process_regression::likelihood(const ParamType & param)
{
	update_kernal(param);

	return lp_param_on_xy();
}

double Causality::gaussian_process_regression::lp_param_on_xy()
{
	//it's log detK
	double lndetK = (ldltK.vectorD().array().abs() + g_paramMin).log().sum();
	double L = 0.5* D *lndetK;

	L += 0.5 * (Y.array() * iKY.array()).sum();

	// Parameter priori
	L += g_paramWeight * log(fabs(alpha()));
	L += g_paramWeight * log(fabs(gamma()));
	L -= g_paramWeight * log(fabs(beta()));

	assert(isfinite(L));
	return L;

}

gaussian_process_regression::ParamType gaussian_process_regression::likelihood_derivative(const ParamType & param)
{
	//if ((param - lparam).cwiseAbs2().sum() > epsilon() * epsilon())
	update_kernal(param);

	return lp_param_on_xy_grad();
}

gpr::ParamType gpr::lp_param_on_xy_grad()
{
	dKgamma = Dx.array() * K.array();

	// L += tr(Y' * iK * Y) = tr(iK * Y * Y') = tr(iK * YY')
	// = sum(iK .* YY') = sum (Y .* iKY)
	iKYYtiK = iKY*iKY.transpose(); // K^-1 * Y * Y' * K^-1 

	 // R = d(L_GP) / d(K)
	R = 0.5f * (D * iK - iKYYtiK);

	ParamType grad(lparam.rows(), lparam.cols());
	// There is the space for improve as tr(A*B) can be simplifed to O(N^2)
	grad[0] = R.cwiseProduct(dKalpha.transpose()).sum() + g_paramWeight / alpha(); // (R * dKa).trace() = sum(R .* dKa') 
	grad[1] = R.trace() - g_paramWeight / beta(); // note, d(K)/d(beta) = I
	grad[2] = R.cwiseProduct(dKgamma.transpose()).sum() + g_paramWeight / gamma();

	return grad;
}

double gaussian_process_regression::optimze_parameters(const ParamType & initial_param)
{
	update_kernal(initial_param);

	struct ParamTuneFunctor : Functor<double>
	{
		ParamTuneFunctor(gaussian_process_regression& gpr)
			:_this(gpr), Functor<double>(ParamType::SizeAtCompileTime, ParamType::SizeAtCompileTime)
		{}
		gaussian_process_regression& _this;
		int operator()(const ColVectorType& param, ValueType& fvec)
		{
			fvec.setZero();
			fvec(0) = _this.likelihood(param);
			return 0;
		}

		int df(const ColVectorType& param, JacobianType& fjac)
		{
			fjac.setZero();
			fjac.col(0) = _this.likelihood_derivative(param);
			return 0;
		}
	} functor(*this);

	//ColVectorType param = initial_param;

	//Eigen::LevenbergMarquardt<ParamTuneFunctor, double> lm(functor);
	//lm.minimize(param);
	//double min_likelihood = lm.fvec(0);

	dlib_vector param(3);
	param(0) = initial_param(0);
	param(1) = initial_param(1);
	param(2) = initial_param(2);

	auto f = std::bind(&dlib_likelihood, this, std::placeholders::_1);
	auto df = std::bind(&dlib_likelihood_derv, this, std::placeholders::_1);

	//auto numberic_diff = dlib::derivative(f)(param);
	//auto anaylatic_diff = df(param);

	//std::cout << "numberic derv = " << dlib::trans(numberic_diff) << "anaylatic derv = " << dlib::trans(anaylatic_diff) << std::endl;
	//std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
	//	<< dlib::length(numberic_diff - anaylatic_diff) << std::endl;

	double min_likelihood = 0;
	try
	{
		min_likelihood =
			dlib::find_min_box_constrained(
				//dlib::find_min(
				dlib::cg_search_strategy(),
				dlib::objective_delta_stop_strategy(1e-4,200),
				f, df,//dlib::derivative(f),//df,
				param, g_paramMin, g_paramMax);

		std::cout << "Minimum of likelihood = " << min_likelihood << ", with alpha = " << param(0) << ", beta = " << param(1) << ", gamma = " << param(2) << std::endl;
	}
	catch (...)
	{
		std::cout << "!!! Fail in optimization" << std::endl;
	}


	lparam(0) = param(0);
	lparam(1) = param(1);
	lparam(2) = param(2);
	update_kernal(lparam);
	return min_likelihood;
}

double gaussian_process_regression::optimze_parameters()
{
	// Use adjactive difference instead of overall varience
	auto adjvarX = sqrtf((X.bottomRows(N - 1) - X.topRows(N - 1)).cwiseAbs2().sum() / (N - 2));
	auto varX = sqrt((X.cwiseAbs2().sum() / (N - 1)));
	assert(!isnan(varX));

	ParamType param;

	std::vector<double> alphas = { 0.05 };
	std::vector<double> betas = { 1e-2, 1e-6};
	std::vector<double> gemmas(5);
	for (size_t i = 0; i < gemmas.size(); i++)
	{
		double t = i / (double)(gemmas.size() - 1);
		gemmas[i] = 1.0 / (adjvarX * (1 - t) + varX * t);
	}

	ParamType bestParam;
	double bestLikelihood = std::numeric_limits<double>::max();

	for (auto alpha : alphas)
	{
		param(0) = alpha;
		for (auto beta : betas)
		{
			param(1) = beta;
			for (auto gemma : gemmas)
			{
				param(2) = gemma;
				double lh = optimze_parameters(param);
				if (lh < bestLikelihood)
				{
					bestLikelihood = lh;
					bestParam = lparam;
				}
			}
		}
	}

	set_parameters(bestParam);


	if (bestLikelihood > 0)
	{
		std::cout << "***** ::::>(>_<)<:::: ***** Parameter is sub-optimal!" << std::endl;
	}

	std::cout << " Var(X) = " << varX << std::endl;


	return bestLikelihood;
}

template <typename DerivedX>
void gplvm::update_kernal(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType& param)
{
	X = x;
	if (parent)
		parent->iKY = iK * X;
	update_Dx();
	gpr::update_kernal(param);
}


template <typename DerivedX>
double gplvm::learning_likelihood(_In_ const Eigen::MatrixBase<DerivedX>& x, const ParamType &param)
{
	update_kernal(x, param);

	//it's log detK
	double L = lp_param_on_xy();

	// Latent variable priori, one-point dynamic can be view as K == K^-1 == I
	if (dyna_type > NoDynamic)
		L += 0.5 * (X.array() * parent->iKY.array()).sum();

	return L;
}

template <typename DerivedXOut, typename DerivedX>
void gplvm::learning_likelihood_derivative(_Out_ Eigen::MatrixBase<DerivedXOut>& dx, _Out_ ParamType& dparam, _In_ const Eigen::MatrixBase<DerivedX>& x, _In_ const ParamType &param)
{
	update_kernal(x, param);

	dparam = lp_param_on_xy_grad();

	// There should be someway to simply the caculation of this using matrix operatorions instead of for
	R.array() *= K.array();
	for (int i = 0; i < N; i++)
	{
		//RKi = (R.row(i).array() * K.row(i).array()).eval();
		auto RKi = R.row(i);
		dx.row(i) = RKi.sum() * X.row(i);
		dx.row(i) -= (RKi.asDiagonal() * X).colwise().sum();
		dx.row(i) *= -2.0 * gamma();
	}

	if (dyna_type > NoDynamic)
		dx += parent->iKY;
}


Eigen::Map<const Eigen::RowVectorXd> as_eigen(const dlib_vector& dlibv)
{
	return Eigen::Map<const Eigen::RowVectorXd>(dlibv.begin(), dlibv.size());
}

Eigen::Map<Eigen::RowVectorXd> as_eigen(dlib_vector& dlibv)
{
	return Eigen::Map<Eigen::RowVectorXd>(dlibv.begin(), dlibv.size());
}

double dib_lik_xparam(gplvm* _this, const dlib_vector& xparam)
{
	gplvm::ParamType param = Eigen::Map<const gplvm::ParamType>(xparam.end() - gplvm::ParamSize, gplvm::ParamSize);

	Eigen::Map<const gplvm::MatrixType> xmap(
		xparam.begin(),
		_this->sample_size(),
		_this->latent_dimension());

	return _this->learning_likelihood(
		xmap,
		param);
}

// This is a helper function used while optimizing the rosen() function.  
dlib_vector dib_lik_xparam_derv(gplvm* _this, const dlib_vector& xparam)
{
	dlib_vector derv(xparam.size());

	Eigen::Map<const gplvm::MatrixType> xmap(
		xparam.begin(),
		_this->sample_size(),
		_this->latent_dimension());

	Eigen::Map<gplvm::MatrixType> dxmap(
		derv.begin(),
		_this->sample_size(),
		_this->latent_dimension());

	gplvm::ParamType dparam, param;
	param = Eigen::Map<const gplvm::ParamType>(xparam.end() - gplvm::ParamSize, gplvm::ParamSize);

	_this->learning_likelihood_derivative(dxmap, dparam,
		xmap,
		param);

	std::copy_n(dparam.data(), 3, derv.end() - 3);

	return derv;
}


double gplvm::learn_model(const ParamType& init_param , Scalar stop_delta , int max_iter )
{
	ParamType initParam = init_param;

	dlib_vector xparam(X.size() + ParamSize);
	dlib_vector param(3);
	std::copy_n(initParam.data(), initParam.size(), param.begin());


	std::copy_n(X.data(), X.size(), xparam.begin());
	std::copy_n(initParam.data(), initParam.size(), xparam.end() - 3);

	auto f = std::bind(&dib_lik_xparam, this, std::placeholders::_1);
	auto df = std::bind(&dib_lik_xparam_derv, this, std::placeholders::_1);
	//auto dpf = std::bind(&dlib_likelihood_derv, static_cast<gpr*>(this), std::placeholders::_1);

	//update_kernal(X, initParam);
	//auto dpf_diff = dpf(param);
	//of << "Pderv=" << dlib::trans(dpf_diff) << std::endl;
	//auto anaylatic_diff = df(xparam);
	//auto numberic_diff = dlib::derivative(f)(xparam);

	//std::ofstream of("lvmderv.txt");
	//of << "Nderv=" << dlib::trans(numberic_diff) << std::endl;
	//of << "Aderv=" << dlib::trans(anaylatic_diff) << std::endl;
	//of.close();

	//std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
	//	<< dlib::length(numberic_diff - anaylatic_diff) << std::endl;

	double min_likelihood = 0;
	try
	{
		min_likelihood =
			dlib::find_min_box_constrained(
				//dlib::find_min(
				dlib::cg_search_strategy(),
				dlib::objective_delta_stop_strategy(stop_delta,max_iter).be_verbose(),
				f, df,//dlib::derivative(f),//df,
				xparam, g_paramMin, g_paramMax);

		std::cout << "Minimum of likelihood = " << min_likelihood << std::endl;

		lparam = Eigen::Map<const gplvm::ParamType>(xparam.end() - gplvm::ParamSize, gplvm::ParamSize);

		Eigen::Map<const gplvm::MatrixType> xmap(
			xparam.begin(),
			gpr::sample_size(),
			this->latent_dimension());

		X = xmap;

		update_kernal(X, lparam);
			//<< ", with alpha = " << param(0) << ", beta = " << param(1) << ", gamma = " << param(2) << std::endl;
	}
	catch (...)
	{
		std::cout << "!!! Fail in optimization" << std::endl;
		_CrtDbgBreak();
	}

	return min_likelihood;
}

// aka. matrixX
inline const gplvm::MatrixType & gplvm::latent_coords() const { return X; }

double gplvm::likelihood_xy(const RowVectorType & x, const RowVectorType & y) const
{
	auto lxy = gpr::get_likelihood_xy(x, y);

	if (dyna_type > NoDynamic)
		lxy += 0.5 * x * parent->iK * x.transpose();

	return lxy;
}

gplvm::RowVectorType gplvm::likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType derv = get_likelihood_xy_derivative(x, y);

	auto dx = derv.segment(0, x.size());
	if (dyna_type > NoDynamic)
		dx += parent->iK * x;

	return derv;
}

void gplvm::initialize(const MatrixType & _Y, Eigen::DenseIndex dX)
{
	N = _Y.rows();
	D = _Y.cols();
	dimX = dX;

	lparam = { 1.0,1.0,1.0 };

	Y = _Y;

	auto pcaY = Eigen::Pca<MatrixType>(Y);
	uY = pcaY.mean();
	Y -= uY.replicate(N, 1);

	X = pcaY.coordinates(dimX);
	uX.setZero(dimX);

	Dx.resize(N, N);
	K.resize(N, N);
	R.resize(N, N);
	iKY.resize(N, Y.cols());
	iKYYtiK.resize(N, N);
	dKx.resize(N, N);

	if (parent)
	{
		parent->K.setIdentity(N, N);
		parent->iK = parent->K;
	}
}

void gaussian_process_lvm::set_dynamic(DynamicTypeEnum type, double timespan, const ParamType * timeparam, ColVectorType * sampleTimes)
{
	dyna_type = type;

	auto& Kt = parent->K;
	auto& tparam = parent->lparam;

	if (timeparam)
		tparam = *timeparam;

	ColVectorType T;
	T.setLinSpaced(0, timespan - timespan / N);
	if (sampleTimes)
		T = *sampleTimes;


	if (dyna_type == OnewayDynamic)
		Kt = (-0.5 * tparam[2]) * (T.replicate(1, N) - T.replicate(1, N).transpose()).array().abs2();
	else if(dyna_type == PeriodicDynamic)
		Kt = (-2.0 * tparam[2]) * ((T.replicate(1, N) - T.replicate(1, N).transpose()).array() * (0.5 / timespan * DirectX::XM_2PI)).sin().abs2();

	Kt = tparam[0] * Kt.array().exp();
	Kt.diagonal().array() += 1 / tparam[1];
	iK.setIdentity(N, N);
	Kt.ldlt().solveInPlace(iK);
}
