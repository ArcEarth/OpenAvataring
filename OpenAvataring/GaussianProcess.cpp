#include "pch.h"
#include "GaussianProcess.h"
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
{}

void gaussian_process_regression::initialize(const Eigen::MatrixXf & _X, const Eigen::MatrixXf & _Y)
{
	assert(_X.rows() == _Y.rows() && _X.rows() > 0 && "Observations count agree");
	assert(!_X.hasNaN());

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
		Dx.row(i) = (-0.5f) * (X.row(i).replicate(N, 1) - X).cwiseAbs2().rowwise().sum().transpose();
	}
}

void gaussian_process_regression::update_kernal(const ParamType & param)
{
	//if ((lparam - param).cwiseAbs().sum() < 1e-8)
	//	return;

	lparam = param;

	int N = X.rows();

	dKalpha.array() = (Dx * gamma()).array().exp();
	K = (alpha() * dKalpha);
	K.diagonal().array() += 1.0 / beta();

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

		ObsvExpectionFunctor(const gaussian_process_regression& gpr,const RowVectorType& _cZ,const Eigen::MatrixXd& _invCov)
			: _this(gpr),cZ(_cZ),invCov(_invCov), Functor<double>(_cZ.size(),1)
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
			double varX = _this.alpha() + 1.0 / _this.beta() - cov;
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
			double varX = _this.alpha() + 1.0 / _this.beta() - cov;

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
	} functor(*this,cZ,invCov);

	auto f = [this, &cZ, &invCov](const dlib_vector& _x) -> double
	{
		auto x = RowVectorType::Map(_x.begin(), _x.end() - _x.begin());

		ColVectorType Kx = (x.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
		Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

		ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
		double cov = Kx.transpose() * iKkx;
		double varX = alpha() + 1.0 / beta() - cov;
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
		double varX = alpha() + 1.0 / beta() - cov;

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

	double varX = alpha() + 1.0 / beta();

	RowVectorType dx = x.cast<double>() - uX;
	ColVectorType Kx = (dx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	varX = abs(varX-cov);

	assert(varX > 0);

	if (y != nullptr)
		*y = uY + iKkx.transpose() * Y;

	return 0.5 * D * log(varX);
}

gaussian_process_regression::ColVectorType gaussian_process_regression::get_expectation_and_likelihood(const MatrixType & x, MatrixType * y) const
{
	assert(x.cols() == X.cols());

	double varX = alpha() + 1.0 / beta();

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
	double varX = alpha() + 1.0 / beta();

	RowVectorType dx = x.cast<double>() - uX;
	ColVectorType Kx = (dx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	varX = std::max(1e-5,varX-cov);

	assert(varX > 0);

	ey = uY + iKkx.transpose() * Y;

	double difY = (y - ey).cwiseAbs2().sum();

	double lxy = 0.5*(difY / varX + D * log(varX));
	return lxy;
}

gaussian_process_regression::RowVectorType gaussian_process_regression::get_likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType ey(y.size());
	double varX = alpha() + 1.0 / beta();

	RowVectorType dx = x.cast<double>() - uX;
	ColVectorType Kx = (dx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	ColVectorType iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	varX = std::max(1e-5,varX-cov);

	assert(varX > 0);

	ey = uY + iKkx.transpose() * Y;

	RowVectorType derv(x.size()+y.size());
	auto difY = (y - ey) / varX;

	derv.segment(x.size(), y.size()) = difY;

	// the derivative to x is complicate.... let's ingnore that term first
	derv.segment(0, x.size()).setZero();

	//double lxy = 0.5*(difY / difX + D * log(difX));
	return derv;
}


double gaussian_process_regression::likelihood(const ParamType & param)
{
	//if ((param - lparam).cwiseAbs2().sum() > epsilon() * epsilon())
	update_kernal(param);

	//it's log detK
	double lndetK = (ldltK.vectorD().array().abs() + g_paramMin).log().sum();
	double L = 0.5* D *lndetK + g_paramWeight * param.array().abs().log().sum();

	// L += tr(Y' * iK * Y) = tr(iK * Y * Y') = tr(iK * YY') = sum(iK .* YY') = sum (Y .* iKY)

	L += 0.5 * (Y.array() * iKY.array()).sum();

	//for (int i = 0; i < D; i++)
	//{
	//	L += 0.5 * Y.col(i).transpose() * iKY.col(i); // Yi' * K^-1 * Yi
	//}

	assert(!isnan(L) && !isinf(L));
	return L;
}

gaussian_process_regression::ParamType gaussian_process_regression::likelihood_derivative(const ParamType & param)
{
	//if ((param - lparam).cwiseAbs2().sum() > epsilon() * epsilon())
	update_kernal(param);

	auto alpha = param[0], beta = param[1], gamma = param[2];

	dKgamma = Dx.array() * K.array();

	iKYYtiK = iKY*iKY.transpose(); // K^-1 * Y * Y' * K^-1 

	//R.setIdentity();
	//R.diagonal() *= D;
	//ldltK.solveInPlace(R); // R = K^-1
	R = iK * D;//

	R -= iKYYtiK;
	R *= 0.5f;

	ParamType grad(param.rows(), param.cols());
	// There is the space for improve as tr(A*B) can be simplifed to O(N^2)
	grad[0] = R.cwiseProduct(dKalpha.transpose()).sum() + g_paramWeight / alpha; // (R * dKa).trace() = sum(R .* dKa') 
	grad[1] = -(R.trace()) / (beta*beta) + g_paramWeight / beta; // note, d(K)/d(beta) = I
	grad[2] = R.cwiseProduct(dKgamma.transpose()).sum() + g_paramWeight / gamma;

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
				dlib::objective_delta_stop_strategy(1e-5),
				f, df,//dlib::derivative(f),//df,
				param, g_paramMin, g_paramMax);

		std::cout << "Minimum of likelihood = " << min_likelihood << ", with alpha = " << param(0) << ", beta = " << param(1) << ", gamma = " << param(2) << std::endl;
	}
	catch(...)
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
	auto adjvarX = sqrtf((X.bottomRows(N - 1) - X.topRows(N - 1)).cwiseAbs2().sum() / (N -2));
	auto varX = sqrt((X.cwiseAbs2().sum() / (N - 1)));
	assert(!isnan(varX));

	ParamType param;

	std::vector<double> alphas = { 0.1, 0.5 , 0.8, 1.0, 1.2, 1.5 };
	std::vector<double> betas = {0.1, 1.0, 10.0};
	std::vector<double> gemmas(10);
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


void gp_lvm::update_kernal(const MatrixType& x, const ParamType& param)
{
	X = x;
	update_Dx();
	gpr::update_kernal(param);
}

double gp_lvm::learning_likelihood(const MatrixType & x, const ParamType & param)
{
	update_kernal(x, param);

	//it's log detK
	double lndetK = (ldltK.vectorD().array().abs() + g_paramMin).log().sum();
	double L = 0.5* D *lndetK + g_paramWeight * param.array().abs().log().sum();

	assert(std::isfinite(L));

	// L += tr(Y' * iK * Y) = tr(iK * Y * Y') = tr(iK * YY') = sum(iK .* YY') = sum (Y .* iKY)

	L += 0.5 * (Y.array() * iKY.array()).sum();

	// NoDynamic
	if (dyna_type == NoDynamic)
		L += 0.5 * x.cwiseAbs2().sum(); // 0.5 * sum(|x_i|^2)

	//for (int i = 0; i < D; i++)
	//{
	//	L += 0.5 * Y.col(i).transpose() * iKY.col(i); // Yi' * K^-1 * Yi
	//}

	return L;
}

void gaussian_process_lvm::learning_likelihood_derivative(MatrixType & dx, ParamType & dparam, const MatrixType & x, const ParamType & param)
{
	update_kernal(x, param);

	//dx = x; // term d(0.5 * sum(|x_i|^2))

	auto alpha = param[0], beta = param[1], gamma = param[2];

	dKgamma = Dx.array() * K.array();

	iKYYtiK = iKY*iKY.transpose(); // K^-1 * Y * Y' * K^-1 

	// R = d(L_GP) / d(K)
	R = D * iK - 0.5f * iKYYtiK;

	// There is the space for improve as tr(A*B) can be simplifed to O(N^2)
	dparam[0] = R.cwiseProduct(dKalpha.transpose()).sum() + g_paramWeight / alpha; // (R * dKa).trace() = sum(R .* dKa') 
	dparam[1] = -(R.trace()) / (beta*beta) + g_paramWeight / beta; // note, d(K)/d(beta) = I
	dparam[2] = R.cwiseProduct(dKgamma.transpose()).sum() + g_paramWeight / gamma;

	// There should be someway to simply the caculation of this using matrix operatorions instead of for
	R.array() *= K.array();
	for (int i = 0; i < N; i++)
	{
		//RKi = (R.row(i).array() * K.row(i).array()).eval();
		auto RKi = R.row(i);
		dx.row(i) = RKi.sum() * X.row(i);
		dx.row(i) -= (RKi.asDiagonal() * X).colwise().sum();
		dx.row(i) *= -2.0 * gamma;
	}

	if (dyna_type == NoDynamic)
		dx += X;
}

double gaussian_process_lvm::likelihood_xy(const RowVectorType & y, const RowVectorType & x)
{
	RowVectorType fx;
	auto varX = get_expectation_and_likelihood(x, &fx);

}
