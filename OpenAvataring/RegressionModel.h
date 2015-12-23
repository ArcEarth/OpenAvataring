#pragma once
#include <Eigen\Core>
#include <memory>

namespace Causality
{
	class IRegression abstract
	{
		// return the confidence of the model
		virtual float Fit(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y) = 0;

		// return the confidence of the prediction
		virtual float Predict(const Eigen::RowVectorXf& X, Eigen::RowVectorXf& Y) = 0;
	};
}