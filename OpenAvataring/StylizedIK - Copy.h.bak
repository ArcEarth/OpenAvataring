#pragma once
#include "Causality\Armature.h"
#include "Causality\Animations.h"
#include "Causality\InverseKinematics.h"
#include "GaussianProcess.h"

namespace Causality
{
	class StylizedChainIK : public ChainInverseKinematics
	{
	public:
		using OptimzeVectorType = Eigen::VectorXf;
		using OptimzeJacobiType = Eigen::MatrixXf;

		class IFeatureDecoder abstract
		{
		public:
			typedef StylizedChainIK::OptimzeVectorType VectorType;
			typedef StylizedChainIK::OptimzeJacobiType JacobiType;

			virtual ~IFeatureDecoder();

			// Decode input vector into local rotation quaternions
			virtual void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) = 0;
			// Encode joint rotations into input vector 
			virtual void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) = 0;
			// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
			virtual void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) = 0;
		};

	private:
		float												m_chainLength;
		gaussian_process_regression							m_gplvm;

		double												m_ikWeight;
		double												m_ikLimitWeight;
		double												m_markovWeight;
		double												m_styleWeight;

		double												m_currentError;

		double												m_meanLk;
		long												m_counter;
		bool												m_cValiad;	// is current validate

		OptimzeVectorType									m_iy;	// initial y, default y
		Eigen::RowVectorXd									m_cx;	// current x
		OptimzeVectorType									m_cy;	// current y

		Eigen::RowVectorXf									m_wy;	//weights of y
		Eigen::MatrixXf										m_limy;	//weights of y

		Eigen::RowVectorXd									m_ey;
		double												m_segmaX;

		std::unique_ptr<IFeatureDecoder>					m_fpDecoder;

	private:
		struct OptimizeFunctor;
		std::unique_ptr<OptimizeFunctor>					m_pFunctor;

		using ChainInverseKinematics::solve;
		using ChainInverseKinematics::solveWithStyle;
	public:
		StylizedChainIK();
		StylizedChainIK(size_t n);
		StylizedChainIK(const std::vector<const Joint*> &joints, ArmatureFrameConstView defaultframe);
		~StylizedChainIK();
		void reset();

		const OptimzeVectorType & apply(const Vector3 & goal, const DirectX::Quaternion & baseRotation);
		const OptimzeVectorType & apply(const Vector3 & goal, const Vector3& goal_vel, const DirectX::Quaternion& baseRotation);
		const OptimzeVectorType & apply(const Eigen::Vector3d & goal, const DirectX::Quaternion & baseRotation)
		{
			return apply(Vector3(goal.x(), goal.y(), goal.z()),baseRotation);
		}
		const OptimzeVectorType & apply(const Eigen::Vector3d & goal, const Eigen::Vector3d& goal_vel, const DirectX::Quaternion & baseRotation)
		{
			return apply(Vector3(goal.x(), goal.y(), goal.z()), Vector3(goal_vel.x(), goal_vel.y(), goal_vel.z()), baseRotation);
		}

		// set the functional that decode feature vector "Y" to local rotation quaternions
		// by default, Decoder is set to "Absolute Ln Quaternion of joint local orientation" 
		// you can use RelativeLnQuaternionDecoder and 
		void setDecoder(std::unique_ptr<IFeatureDecoder>&& decoder);
		const IFeatureDecoder* getDecoder() const { return m_fpDecoder.get(); }
		IFeatureDecoder* getDecoder() { return m_fpDecoder.get(); }

		void setChain(const std::vector<const Joint*> &joints, ArmatureFrameConstView defaultframe);
		void setIKWeight(double weight);
		void setMarkovWeight(double weight);

		template <class Derived>
		void setGplvmWeight(const Eigen::DenseBase<Derived>& w) { m_wy = w; }

		template <class Derived>
		void setYLimit(const Eigen::DenseBase<Derived>& limY) { m_limy = limY; }

		gaussian_process_regression& Gplvm() { return m_gplvm; }
		const gaussian_process_regression& Gplvm() const { return m_gplvm; }
	};

	class AbsoluteLnQuaternionDecoder : public StylizedChainIK::IFeatureDecoder
	{
	public:
		~AbsoluteLnQuaternionDecoder();
		// Decode input vector into local rotation quaternions
		void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
		// Encode joint rotations into input vector 
		void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
		// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
		void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	};

	class AbsoluteEulerAngleDecoder : public StylizedChainIK::IFeatureDecoder
	{
	public:
		~AbsoluteEulerAngleDecoder();
		// Decode input vector into local rotation quaternions
		void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
		// Encode joint rotations into input vector 
		void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
		// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
		void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	};

	class RelativeEulerAngleDecoder : public AbsoluteEulerAngleDecoder
	{
	public:
		std::vector<DirectX::Quaternion, DirectX::XMAllocator>
			bases;
	public:
		~RelativeEulerAngleDecoder();
		// Decode input vector into local rotation quaternions
		void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
		// Encode joint rotations into input vector 
		void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
		// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
		void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	};

	class RelativeLnQuaternionDecoder : public AbsoluteLnQuaternionDecoder
	{
	public:
		std::vector<DirectX::Quaternion, DirectX::XMAllocator>
			bases;
	public:
		~RelativeLnQuaternionDecoder();
		// Decode input vector into local rotation quaternions
		void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
		// Encode joint rotations into input vector 
		void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
		// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
		void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	};

	class RelativeEulerAnglePcaDecoder : public RelativeEulerAngleDecoder
	{
	public:
		Eigen::RowVectorXd	meanY;
		Eigen::MatrixXd		pcaY;
		Eigen::MatrixXd		invPcaY;
	public:
		~RelativeEulerAnglePcaDecoder();
		// Decode input vector into local rotation quaternions
		void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
		// Encode joint rotations into input vector 
		void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
		// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
		void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	};

	class RelativeLnQuaternionPcaDecoder : public RelativeLnQuaternionDecoder
	{
	public:
		Eigen::RowVectorXd	meanY;
		Eigen::MatrixXd		pcaY;
		Eigen::MatrixXd		invPcaY;
	public:
		~RelativeLnQuaternionPcaDecoder();
		// Decode input vector into local rotation quaternions
		void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
		// Encode joint rotations into input vector 
		void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
		// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
		void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	};


#define AUTO_PROPERTY(type,name) private : type m_##name;\
public:\
	type& name() { return m_##name; } \
	const type& name() { return m_##name; } \
	void set_##name(const type& val) { m_##name = val;}

}