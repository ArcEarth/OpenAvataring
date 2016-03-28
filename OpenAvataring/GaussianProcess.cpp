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

// if there is no noticable difference, than we should discard this dimension
static constexpr double g_input_scale_epsilon = 1e-3;
// the anisometric scale should not be greater than this
static constexpr double g_input_scaleup_limit = 10.0;
static constexpr double almost_zero = 1e-6;
static constexpr double g_paramMin = 1e-3;
static constexpr double g_paramMax = 1e3;
static constexpr double g_paramWeight = 1;

typedef dlib::matrix<double, 0, 1> dlib_vector;

gaussian_process_regression::gaussian_process_regression(const Eigen::MatrixXf & _X, const Eigen::MatrixXf & _Y)
//: DenseFunctor<float, 3, 1>()
{
	initialize(_X, _Y);
}

gaussian_process_regression::gaussian_process_regression()
{
	dimX = 0; N = 0; dimY = 0;
}

void gaussian_process_regression::initialize(Index _N, Index _DimY, Index _DimX)
{
	dimX = _DimX; N = _N; dimY = _DimY;
	lparam.setZero(ParamSize);
	wY.setOnes(dimY);
	wX.setOnes(dimX);

	X.resize(N, dimX);
	Y.resize(N, dimY);
	K.resize(N, N);

	uX.setZero(dimX);
	uY.setZero(dimY);
	Dx.resize(N, N);
	R.resize(N, N);
	RcK.resize(N, N);
	iKY.resize(N, dimY);
	iK.resize(N, N);

	iKYYtiK.resize(N, N);
	dKalpha.resize(N, N);
	dKgamma.resize(N, N);
}

void gaussian_process_regression::initialize(const Eigen::MatrixXf & _X, const Eigen::MatrixXf & _Y)
{
	assert(_X.rows() == _Y.rows() && _X.rows() > 0 && "Observations count agree");
	assert(!_X.hasNaN());

	dimX = _X.cols();
	N = _X.rows();
	dimY = _Y.cols();
	initialize(N, dimY, dimX);

	X = _X.cast<double>();
	Y = _Y.cast<double>();
	uX = X.colwise().mean();
	uY = Y.colwise().mean();
	X = X - uX.replicate(N, 1);
	Y = Y - uY.replicate(N, 1);

	// Normalize X into uniform gaussian
	int nsam = N > 1 ? N - 1 : N;
	double stdevX = sqrt(X.cwiseAbs2().sum() / nsam);
	// anisometric scale to stdev
	wX = (X.cwiseAbs2().colwise().sum() / nsam).cwiseSqrt() / stdevX;

	wX = (wX.array() > g_input_scale_epsilon).select(wX, 1.0)
		.cwiseInverse().cwiseMin(g_input_scaleup_limit) / stdevX;

	X = X * wX.asDiagonal();

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
	lparam = param;

	dKalpha.array() = (Dx * gamma()).array().exp();
	K = (alpha() * dKalpha);
	K.diagonal().array() += beta();

	ldltK.compute(K);
	assert(ldltK.info() == Eigen::Success);

	iK = ldltK.solve(Eigen::MatrixXd::Identity(N, N));
	iKY.noalias() = iK * Y;//ldltK.solve(Y);
}

double gaussian_process_regression::get_ey_on_obser_x(const RowVectorType & z, _In_ const MatrixType &covXZ, RowVectorType * y) const
{
	// Center Z with the mean
	assert(z.size() == covXZ.rows() && covXZ.rows() == covXZ.cols() && "CovXZ should be positive semi-defined");
	RowVectorType cZ = z - uX;
	auto ldltCov = covXZ.ldlt();
	auto invCov = ldltCov.solve(MatrixType::Identity(z.size(), z.size())).eval();
	auto detCov = ldltCov.vectorD().prod();

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
		double value = 0.5 * dimY * log(varX) + 0.5 * dxCz * invCov * dxCz.transpose();

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
		derv = -dimY / varX * derv + (x - cZ) * invCov; // / varZ;

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

	double likelihood = dlib::find_min(
		dlib::cg_search_strategy(),
		dlib::objective_delta_stop_strategy(1e-3),
		f, df,//dlib::derivative(f)
		_x,
		std::numeric_limits<double>::min());

	RowVectorType eX = RowVectorType::Map(_x.begin(), _x.end() - _x.begin()) + uX;

	get_ey_on_x(eX, y);

	likelihood = exp(-likelihood);

	return likelihood;
}

void gpr::lp_xy_helper(const RowVectorType & x, RowVectorType & zx, ColVectorType &Kx, ColVectorType &iKkx, double &varX, RowVectorType &ey) const
{
	// Normalize x
	zx = x - uX;
	zx = zx.array() * wX.array();

	Kx = (zx.replicate(N, 1) - X).cwiseAbs2().rowwise().sum();
	Kx.array() = ((-0.5*gamma()) * Kx.array()).exp() * alpha();

	iKkx = iK * Kx;//ldltK.solve(Kx);
	double cov = Kx.transpose() * iKkx;
	varX = alpha() + beta() - cov;
	// ensure the result is valiad(not nessary correct) under numberic jittering close to zero
	varX = fmax(fabs(varX), almost_zero);

	ey = uY + iKkx.transpose() * Y;
}

double gaussian_process_regression::get_ey_on_x(const RowVectorType & x, RowVectorType * y) const
{
	RowVectorType ey(dimY) ,zx(dimX);
	ColVectorType Kx(N), iKkx(N);
	double varX;

	lp_xy_helper(x, zx, Kx, iKkx, varX, ey);

	if (y != nullptr)
		*y = ey;

	return 0.5 * dimY * log(fabs(varX));
}

gaussian_process_regression::ColVectorType gaussian_process_regression::get_ey_on_x(const MatrixType & x, MatrixType * y) const
{
	assert(x.cols() == X.cols());

	double varX = alpha() + beta();

	MatrixType zx = x - uX.replicate(x.rows(), 1);
	zx = zx * wX.asDiagonal();

	MatrixType Kx(N, x.rows());
	for (size_t i = 0; i < x.rows(); i++)
		Kx.col(i) = (-0.5*gamma()) * (zx.row(i).replicate(N, 1) - X).cwiseAbs2().rowwise().sum();

	Kx = Kx.array().exp() * alpha(); // N x m, m = x.rows()

	MatrixType iKkx = iK * Kx;//ldltK.solve(Kx); // N x m

	if (y != nullptr)
		*y = uY.replicate(x.rows(), 1) + iKkx.transpose() * Y;

	ColVectorType cov = (Kx.array() * iKkx.array()).colwise().sum().transpose();
	cov = (0.5 * dimY) * (varX - cov.array()).abs().log();

	return cov;
}

// negitive log likilihood of P(y | theta,x)
double gaussian_process_regression::get_likelihood_xy(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType ey(dimY), zx(dimX);
	ColVectorType Kx(N), iKkx(N);
	double varX;

	lp_xy_helper(x, zx, Kx, iKkx, varX, ey);

	double difY = (y - ey).squaredNorm();

	double lxy = 0.5 / varX * difY  + 0.5 * dimY * log(varX);

	return lxy;
	// return 0.5 * difY;
	// return Kx.squaredNorm();
	// return 1.0 / varX;
	// return log(varX);
}


gaussian_process_regression::RowVectorType gaussian_process_regression::get_likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType ey(dimY), zx(dimX);
	ColVectorType Kx(N), iKkx(N);
	double varX;

	lp_xy_helper(x, zx, Kx, iKkx, varX, ey);

	RowVectorType derv(dimX + dimY);
	auto difY = (y - ey).eval();

	auto dy = derv.segment(dimX, dimY);
	dy = difY / varX;
	double sqndify = difY.squaredNorm();

	// after this line , all d(XXX) is d(XXX)/dx
	auto dKx = (zx.replicate(N, 1) - X).eval();
	dKx = -gamma() * Kx.asDiagonal() * dKx;

	// dfx = Y' * K^-1 * dKx = (K^-1' * Y)' * dKx = (K^-1 * Y)' * dKx
	auto dfx = (iKY.transpose() * dKx).eval();

	auto dvarx = (-2 * Kx.transpose() * iK * dKx).eval();

	auto dx = derv.segment(0, dimX);
	//dx = (-0.5/varX) * difY * dfx + (/*dimY*/ - difY.squaredNorm() / varX ) / (2 * varX) * dvarx;
	//dx = 0.5 * ( - difY * dfx / varX - difY.squaredNorm() / (varX * varX) * dvarx);
	dx = -0.5 * sqndify / (varX*varX) * dvarx - difY * dfx / varX + 0.5 * dimY * dvarx / varX;
	// ****** dKx && dvarx : Checked
	// dx = -difY * dfx; // lxy = 0.5 * difY.squaredNorm();
	// dx = 2.0 * Kx.transpose() * dKx;
	// dx = -1.0/(varX*varX) * dvarx; // lxy = 1.0/ varX;
	// dx = dvarx / varX; // log(varX);
	// dy.setZero();
	return derv;
}


double gaussian_process_regression::learning_likelihood_on_xy(const ParamType & param)
{
	update_kernal(param);
	return lp_param_on_xy();
}

double Causality::gaussian_process_regression::lp_param_on_xy()
{
	//it's log detK
	double lndetK = (ldltK.vectorD().array().abs().cwiseMax(almost_zero)).log().sum();
	double L = 0.5* dimY *lndetK;

	L += 0.5 * (Y.array() * iKY.array()).sum();

	// Parameter priori
	L += g_paramWeight * log(fabs(alpha()));
	L += g_paramWeight * log(fabs(gamma()));
	L -= g_paramWeight * log(fabs(beta()));

	assert(isfinite(L));
	return L;

}

gaussian_process_regression::ParamType gaussian_process_regression::learning_likelihood_on_xy_derivative(const ParamType & param)
{
	update_kernal(param);

	return lp_param_on_xy_grad();
}

gpr::ParamType gpr::lp_param_on_xy_grad()
{
	dKgamma = Dx.array() * K.array();

	// L += tr(Y' * iK * Y) = tr(iK * Y * Y') = tr(iK * YY') = sum(iK .* YY') = sum (Y .* iKY)
	iKYYtiK = iKY*iKY.transpose(); // K^-1 * Y * Y' * K^-1 

	 // R = d(L_GP) / d(K)
	R = 0.5f * (dimY * iK - iKYYtiK);

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

	dlib_vector param(3);
	param(0) = initial_param(0);
	param(1) = initial_param(1);
	param(2) = initial_param(2);

	auto f = [this](const dlib_vector& m) -> double
	{
		gaussian_process_regression::ParamType param(m(0), m(1), m(2));
		return this->learning_likelihood_on_xy(param);
	};

	auto df = [this](const dlib_vector& m) -> dlib_vector
	{
		gaussian_process_regression::ParamType param(m(0), m(1), m(2));
		auto grad = this->learning_likelihood_on_xy_derivative(param);

		dlib_vector result = m;
		result(0) = grad(0);
		result(1) = grad(1);
		result(2) = grad(2);
		return result;
	};

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

	std::vector<double> alphas = { 1.0 };
	std::vector<double> betas = { 10.0, 1.0 };
	std::vector<double> gemmas(2);
	for (size_t i = 0; i < gemmas.size(); i++)
	{
		double t = i / (double)(gemmas.size() - 1);
		gemmas[i] = varX / (adjvarX * (1 - t) + varX * t);
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
	gpr::update_kernal<DerivedX>(x,param);
	if (parent)
		parent->iKY = iK * X;
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
	RcK = R.array() * K.array();
	for (int i = 0; i < N; i++)
	{
		//RKi = (R.row(i).array() * K.row(i).array()).eval();
		auto RKi = RcK.row(i);
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

double gplvm::learn_model(const ParamType& init_param , Scalar stop_delta , int max_iter )
{
	ParamType initParam = init_param;

	{
		Eigen::Pca<MatrixType> pcaY(Y);
		X = pcaY.coordinates(dimX);
	}

	dlib_vector xparam(X.size() + ParamSize);
	dlib_vector param(3);
	std::copy_n(initParam.data(), initParam.size(), param.begin());
	std::copy_n(X.data(), X.size(), xparam.begin());
	std::copy_n(initParam.data(), initParam.size(), xparam.end() - 3);

	auto f = [this](const dlib_vector& xparam) -> double
	{
		gplvm::ParamType param = Eigen::Map<const gplvm::ParamType>(xparam.end() - gplvm::ParamSize, gplvm::ParamSize);

		Eigen::Map<const gplvm::MatrixType> xmap(
			xparam.begin(),
			this->sample_size(),
			this->latent_dimension());

		return this->learning_likelihood(
			xmap,
			param);
	};

	auto df = [this](const dlib_vector& xparam) -> dlib_vector
	{
		dlib_vector derv(xparam.size());

		Eigen::Map<const gplvm::MatrixType> xmap(
			xparam.begin(),
			this->sample_size(),
			this->latent_dimension());

		Eigen::Map<gplvm::MatrixType> dxmap(
			derv.begin(),
			this->sample_size(),
			this->latent_dimension());

		gplvm::ParamType dparam, param;
		param = Eigen::Map<const gplvm::ParamType>(xparam.end() - gplvm::ParamSize, gplvm::ParamSize);

		this->learning_likelihood_derivative(dxmap, dparam,
			xmap,
			param);

		std::copy_n(dparam.data(), 3, derv.end() - 3);

		return derv;
	};
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
				dlib::objective_delta_stop_strategy(stop_delta,max_iter),//.be_verbose(),
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


double gplvm::learn_model(int max_iter)
{
	// Use adjactive difference instead of overall varience
	auto adjvarX = sqrtf((X.bottomRows(N - 1) - X.topRows(N - 1)).cwiseAbs2().sum() / (N - 2));
	auto varX = sqrt((X.cwiseAbs2().sum() / (N - 1)));
	assert(!isnan(varX));

	ParamType param;

	std::vector<double> alphas = { 0.05,0.5, 1.0 };
	std::vector<double> betas = { 1.0, 1e-2, 1e-4 };
	std::vector<double> gemmas(5);
	for (size_t i = 0; i < gemmas.size(); i++)
	{
		double t = i / (double)(gemmas.size() - 1);
		gemmas[i] = 1.0 / (adjvarX * (1 - t) + varX * t);
	}

	ParamType bestParam;
	MatrixType bestX;
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
				double lh = learn_model(param, 1e-2, max_iter);
				if (lh < bestLikelihood)
				{
					bestLikelihood = lh;
					bestParam = lparam;
					bestX = X;
				}
			}
		}
	}

	update_kernal(bestX, bestParam);

	if (bestLikelihood > 0)
	{
		std::cout << "***** ::::>(>_<)<:::: ***** Parameter is sub-optimal!" << std::endl;
	}

	std::cout << " Var(X) = " << varX << std::endl;


	return bestLikelihood;
}


double gplvm::get_likelihood_xy(const RowVectorType & x, const RowVectorType & y) const
{
	auto lxy = gpr::get_likelihood_xy(x, y);

	if (dyna_type > NoDynamic)
		lxy += 0.5 * x * parent->iK * x.transpose();

	return lxy;
}

gplvm::RowVectorType gplvm::get_likelihood_xy_derivative(const RowVectorType & x, const RowVectorType & y) const
{
	RowVectorType derv = gpr::get_likelihood_xy_derivative(x, y);

	auto dx = derv.segment(0, x.size());
	if (dyna_type > NoDynamic)
		dx += parent->iK * x;

	return derv;
}

void gplvm::initialize(const MatrixType & _Y, Eigen::DenseIndex dX)
{
	initialize(_Y.rows(), _Y.cols(), dX);

	lparam = { 1.0,1.0,1.0 };

	auto pcaY = Eigen::Pca<MatrixType>(_Y);
	uY = pcaY.mean();
	Y = _Y - uY.replicate(N, 1);

	X = pcaY.coordinates(dimX);
	uX.setZero(dimX);

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

double gaussian_process_lvm::load_model(const MatrixType & _X, const ParamType & _param)
{
	update_kernal(_X, _param);
	lp_param_on_xy_grad(); // Update YiKKtY, R, etc...
	return learning_likelihood(_X, _param);
	//YtiK = Y.transpose() * iK;
}
