#include "pch.h"
#include "StylizedIK.h"

using namespace Causality;
using namespace std;
using namespace Eigen;
using namespace DirectX;

namespace Causality
{
	void AbsoluteLnQuaternionDecode(_Out_cap_(n) DirectX::Quaternion* rots, const Eigen::RowVectorXd& y)
	{
		int n = y.size() / 3;
		Eigen::Vector4f qs;
		XMVECTOR q;
		qs.setZero();
		for (int i = 0; i < n; i++)
		{
			qs.segment<3>(0) = y.segment<3>(i * 3).cast<float>();
			q = XMLoadFloat4A(qs.data());
			q = XMQuaternionExp(q); // revert the log map
			XMStoreA(rots[i], q);
		}
	}


	RelativeLnQuaternionDecoder::~RelativeLnQuaternionDecoder()
	{}
	AbsoluteLnQuaternionDecoder::~AbsoluteLnQuaternionDecoder()
	{}
	RelativeLnQuaternionPcaDecoder::~RelativeLnQuaternionPcaDecoder()
	{}

	void AbsoluteLnQuaternionDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
	{
		int n = rots.size();

		Eigen::Vector4f qs;
		XMVECTOR q;
		qs.setZero();
		for (int i = 0; i < n; i++)
		{
			qs.segment<3>(0) = x.segment<3>(i * 3).cast<float>();
			q = XMLoadFloat4A(qs.data());
			q = XMQuaternionExp(q); // revert the log map
			XMStoreA(rots[i], q);
		}
	}

	void AbsoluteLnQuaternionDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
	{
	}

	void AbsoluteLnQuaternionDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
	{}

	void RelativeLnQuaternionDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
	{
		int n = rots.size();
		Eigen::Vector4f qs;
		XMVECTOR q, qb;
		qs.setZero();
		for (int i = 0; i < n; i++)
		{
			qs.segment<3>(0) = x.segment<3>(i * 3).cast<float>();
			q = XMLoadFloat4A(qs.data());
			q = XMQuaternionExp(q); // revert the log map
			qb = XMLoadA(bases[i]);
			q = XMQuaternionMultiply(qb, q);
			XMStoreA(rots[i], q);
		}
	}

	void RelativeLnQuaternionDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
	{
	}

	void RelativeLnQuaternionDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
	{
	}

	void RelativeLnQuaternionPcaDecoder::Decode(array_view<DirectX::Quaternion> rots, const VectorType & x)
	{
		VectorType dy = (x.cast<double>() * invPcaY + meanY).cast<float>();
		RelativeLnQuaternionDecoder::Decode(rots, dy);
	}

	void RelativeLnQuaternionPcaDecoder::Encode(array_view<const DirectX::Quaternion> rots, VectorType & x)
	{
		RelativeLnQuaternionDecoder::Encode(rots, x);
		RowVectorXd dx = x.transpose().cast<double>();
		dx -= meanY;
		dx *= pcaY;
		x = dx.transpose().cast<float>();
	}

	void RelativeLnQuaternionPcaDecoder::EncodeJacobi(array_view<const DirectX::Quaternion> rotations, JacobiType & jacb)
	{
		jacb *= pcaY;
	}

	StylizedChainIK::IFeatureDecoder::~IFeatureDecoder()
	{
	}

}