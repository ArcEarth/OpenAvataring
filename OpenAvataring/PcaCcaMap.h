#pragma once
#include "CCA.h"

namespace Causality
{
	struct CcaMap
	{
		Eigen::DenseIndex Jx, Jy;
		Eigen::MatrixXf A, B;
		Eigen::RowVectorXf uX, uY;
		Eigen::JacobiSVD<Eigen::MatrixXf> svdBt;
		Eigen::MatrixXf invB;
		bool useInvB;

		// Get the homogenens transform matrix (X.rows + 1) x (Y.rows + 1)
		Eigen::MatrixXf TransformMatrix() const;


		template <class DerivedX, class DerivedY>
		void Apply(_In_ const Eigen::DenseBase<DerivedX> &Xp, _Out_ Eigen::DenseBase<DerivedY> &Yp) const;

		float CreateFrom(_In_ const Eigen::MatrixXf &X, const _In_ Eigen::MatrixXf &Y);
	};

	struct PcaCcaMap : public CcaMap
	{
		Eigen::MatrixXf	pcX, pcY; // Principle components of X or Y
		Eigen::RowVectorXf uXpca, uYpca; // Mean of X or Y

		// Get the homogenens transform matrix (X.rows + 1) x (Y.rows + 1)
		Eigen::MatrixXf TransformMatrix() const;

		template <class DerivedX, class DerivedY>
		void Apply(_In_ const Eigen::DenseBase<DerivedX> &Xp, _Out_ Eigen::MatrixBase<DerivedY> &Yp) const;

		float CreateFrom(_In_ const Eigen::MatrixXf &X, const _In_ Eigen::MatrixXf &Y, float Xcutoff = 0.04f, float Ycutoff = 0.04f);
	};

	template <class DerivedX, class DerivedY>
	inline void ApplyCcaMap(_In_ const CcaMap& map, _In_ const Eigen::DenseBase<DerivedX> &Xp, _Out_ Eigen::DenseBase<DerivedY> &Yp)
	{
		auto U = ((Xp.rowwise() - map.uX) * map.A).eval();
		if (map.useInvB)
			Yp = U * map.invB;
		else
			Yp = map.svdBt.solve(U.transpose()).transpose(); // Y' ~ B' \ U'
		Yp.rowwise() += map.uY;
	}

	template <class DerivedX, class DerivedY>
	inline void ApplyPcaCcaMap(_In_ const PcaCcaMap& map, _In_ const Eigen::DenseBase<DerivedX> &Xp, _Out_ Eigen::MatrixBase<DerivedY> &Yp)
	{
		using namespace Eigen;
		// Project X with PCA
		auto Xpca = ((Xp.rowwise() - map.uXpca) * map.pcX).eval();
		// Project Xpca with CCA to latent space
		auto U = ((Xpca.rowwise() - map.uX) * map.A).eval();
		// Recover Y from latent space
		Matrix<DenseBase<DerivedX>::Scalar, -1, -1> Y;
		if (map.useInvB)
			Y = U * map.invB;
		else
			Y = map.svdBt.solve(U.transpose()).transpose(); // Y' ~ B' \ U'
															// Add the mean
		Y.rowwise() += map.uY;
		// Reconstruct by principle components
		Yp = Y * map.pcY.transpose();
		Yp.rowwise() += map.uYpca;
	}

	template <class DerivedX, class DerivedY>
	inline void CcaMap::Apply(_In_ const Eigen::DenseBase<DerivedX> &Xp, _Out_ Eigen::DenseBase<DerivedY> &Yp) const
	{
		ApplyCcaMap<DerivedX, DerivedY>(*this, Xp, Yp);
	}

	template <class DerivedX, class DerivedY>
	inline void PcaCcaMap::Apply(_In_ const Eigen::DenseBase<DerivedX> &Xp, _Out_ Eigen::MatrixBase<DerivedY> &Yp) const
	{
		ApplyPcaCcaMap<DerivedX, DerivedY>(*this, Xp, Yp);
	}

	inline float CreateCcaMap(CcaMap& map, _In_ const Eigen::MatrixXf &X, const _In_ Eigen::MatrixXf &Y)
	{
		using namespace Eigen;

		Eigen::QrStore<Eigen::MatrixXf> qrX(X), qrY(Y);
		Eigen::Cca<float> cca;
		cca.computeFromQr(qrX, qrY, true);

		if (cca.rank() == 0) return .0f;

		map.Jx = 0; map.Jy = 0;
		map.A = cca.matrixA();
		map.B = cca.matrixB();
		map.uX = qrX.mean();
		map.uY = qrY.mean();

		if (cca.rank() == qrY.cols()) // d == dY
		{
			map.useInvB = true;
			map.invB = map.B.inverse();
		}
		else
		{
			map.useInvB = false;
			map.svdBt = Eigen::JacobiSVD<Eigen::MatrixXf>(map.B.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);;
		}
		return cca.correlaltions().minCoeff();
	}

	inline float CreatePcaCcaMap(PcaCcaMap& map, _In_ const Eigen::MatrixXf &X, _In_ const Eigen::MatrixXf &Y, float Xcutoff = 0.04f, float Ycutoff = 0.04f)
	{
		using namespace Eigen;
		Pca<MatrixXf> pcaX(X), pcaY(Y);
		map.uXpca = pcaX.mean();
		map.uYpca = pcaY.mean();

		auto dX = pcaX.reducedRank(Xcutoff);
		auto dY = pcaY.reducedRank(Ycutoff);
		map.pcX = pcaX.components(dX);
		map.pcY = pcaY.components(dY);

		return CreateCcaMap(map, pcaX.coordinates(dX), pcaY.coordinates(dY));
	}

	inline float CcaMap::CreateFrom(_In_ const Eigen::MatrixXf &X, const _In_ Eigen::MatrixXf &Y)
	{
		return CreateCcaMap(*this, X, Y);
	}

	inline Eigen::MatrixXf CcaMap::TransformMatrix() const
	{
		auto dX = uX.size(), dY = uY.size();
		Eigen::MatrixXf T(dX + 1, dY + 1);

		auto r = T.block(0, 0, dX, dY);

		if (useInvB)
			r.noalias() = A * invB;
		else
			r.noalias() = svdBt.solve(r.transpose()).transpose();

		// translation
		T.block(dX, 0, 1, dY).noalias() = uY - uX * r;

		T(dX, dY) = 1.0f;

		return T;
	}

	inline Eigen::MatrixXf PcaCcaMap::TransformMatrix() const
	{
		//assert(!"Not implented yet, uXpca, uYpca is not considered yet");
		auto dX = uXpca.size(), dY = uYpca.size();
		Eigen::MatrixXf T(dX + 1, dY + 1);
		Eigen::MatrixXf ri;

		auto r = T.block(0, 0, dX, dY);
		auto t = T.block(dX, 0, 1, dY);

		if (useInvB)
			ri.noalias() = A * invB;
		else
			ri.noalias() = svdBt.solve(A.transpose()).transpose();

		r.noalias() = pcX * ri * pcY.transpose(); // pcX, pcY is orthgonal

		// translation
		t.noalias() = (uY - uX * ri) * pcY.transpose() + uYpca - uXpca * r;

		T(dX, dY) = 1.0f;

		return T;
	}

	inline float PcaCcaMap::CreateFrom(_In_ const Eigen::MatrixXf &X, const _In_ Eigen::MatrixXf &Y, float Xcutoff, float Ycutoff)
	{
		return CreatePcaCcaMap(*this, X, Y, Xcutoff, Ycutoff);
	}
}