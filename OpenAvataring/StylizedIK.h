#pragma once
#include <Causality\Armature.h>
#include <Causality\Animations.h>

//#include "Causality\InverseKinematics.h"
#include "GaussianProcess.h"

namespace Causality
{
	class StylizedChainIK //: public ChainInverseKinematics
	{
	public:
		using OptimzeVectorType = Eigen::RowVectorXf;
		using OptimzeJacobiType = Eigen::MatrixXd;

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
		// translation of local chain, 128 bit aligned
		std::vector<DirectX::Vector4, DirectX::XMAllocator>	m_chain;
		float												m_chainLength;

		// intermediate variable for IK caculation
		std::vector<DirectX::Quaternion, DirectX::XMAllocator>
			m_chainRot;

		gaussian_process_regression							m_gpr;
		gaussian_process_lvm								m_gplvm;

		double												m_ikWeight;
		double												m_ikLimitWeight;
		double												m_markovWeight;
		double												m_styleWeight;

		double												m_currentError;

		double												m_meanLk;
		long												m_counter;
		bool												m_cValiad;	// is current validate
		Eigen::RowVectorXd									m_iy;	// initial y, default y
		Eigen::RowVectorXd									m_cx;	// current x
		Eigen::RowVectorXd									m_cy;	// current y
		Eigen::RowVectorXd									m_cyNorm;
		Eigen::RowVectorXd									m_eyNorm;
		Eigen::RowVectorXd									m_wy;	//weights of y
		Eigen::MatrixXd										m_limy;	//weights of y

		Eigen::RowVectorXd									m_ey;
		double												m_segmaX;

		Eigen::Vector3d										m_goal;
		Quaternion											m_baseRot;
		int													m_maxIter;

		std::unique_ptr<IFeatureDecoder>					m_fpDecoder;

	//private:
	//	struct OptimizeFunctor;
	//	std::unique_ptr<OptimizeFunctor>					m_pFunctor;

		//using ChainInverseKinematics::solve;
		//using ChainInverseKinematics::solveWithStyle;
	public:
		StylizedChainIK();
		StylizedChainIK(size_t n);
		StylizedChainIK(const std::vector<const Joint*> &joints, ArmatureFrameConstView defaultframe);
		~StylizedChainIK();
		void reset();

		//const OptimzeVectorType & apply(const Vector3 & goal, const DirectX::Quaternion & baseRotation);
		//const OptimzeVectorType & apply(const Vector3 & goal, const Vector3& goal_vel, const DirectX::Quaternion& baseRotation);
		Eigen::RowVectorXd apply(const Eigen::Vector3d & goal, const DirectX::Quaternion & baseRotation);
		//{
		//	return apply(Vector3(goal.x(), goal.y(), goal.z()),baseRotation);
		//}
		Eigen::RowVectorXd apply(const Eigen::Vector3d & goal, const Eigen::Vector3d& goal_vel, const DirectX::Quaternion & baseRotation);
		//{
		//	return apply(Vector3(goal.x(), goal.y(), goal.z()), Vector3(goal_vel.x(), goal_vel.y(), goal_vel.z()), baseRotation);
		//}
		Eigen::RowVectorXd apply(const Eigen::Vector3d & goal, const Eigen::VectorXd & hint_y);


		// set the functional that decode feature vector "Y" to local rotation quaternions
		// by default, Decoder is set to "Absolute Ln Quaternion of joint local orientation" 
		// you can use RelativeLnQuaternionDecoder and 
		void setDecoder(std::unique_ptr<IFeatureDecoder>&& decoder);
		const IFeatureDecoder* getDecoder() const { return m_fpDecoder.get(); }
		IFeatureDecoder* getDecoder() { return m_fpDecoder.get(); }

		void setGoal(const Eigen::Vector3d & goal);
		void setBaseRotation(const DirectX::Quaternion & q);
		void setHint(const Eigen::RowVectorXd & y);

		void setChain(const std::vector<const Joint*> &joints, ArmatureFrameConstView defaultframe);
		void setIKWeight(double weight);
		void setMarkovWeight(double weight);

		template <class Derived>
		void setGplvmWeight(const Eigen::DenseBase<Derived>& w) { m_wy = w; }

		template <class Derived>
		void setYLimit(const Eigen::DenseBase<Derived>& limY) { m_limy = limY; }

		gaussian_process_regression& Gpr() { return m_gpr; }
		const gaussian_process_regression& Gpr() const { return m_gpr; }

		gplvm& Gplvm() { return m_gplvm; }
		const gplvm& Gplvm() const { return m_gplvm; }

		double objective(const Eigen::RowVectorXd &x, const Eigen::RowVectorXd &y);

		Eigen::RowVectorXd objective_derv(const Eigen::RowVectorXd & x, const Eigen::RowVectorXd & y);

		DirectX::XMVECTOR EndPosition(const DirectX::XMFLOAT4A* rotqs);
		Eigen::Matrix3Xf EndPositionJacobi(const DirectX::XMFLOAT4A* rotqs);

		void JacobbiFromR(DirectX::XMFLOAT4X4A &jac, _In_reads_(3) const float* r);

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

	//class AbsoluteEulerAngleDecoder : public StylizedChainIK::IFeatureDecoder
	//{
	//public:
	//	~AbsoluteEulerAngleDecoder();
	//	// Decode input vector into local rotation quaternions
	//	void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
	//	// Encode joint rotations into input vector 
	//	void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
	//	// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
	//	void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	//};

	//class RelativeEulerAngleDecoder : public AbsoluteEulerAngleDecoder
	//{
	//public:
	//	std::vector<DirectX::Quaternion, DirectX::XMAllocator>
	//		bases;
	//public:
	//	~RelativeEulerAngleDecoder();
	//	// Decode input vector into local rotation quaternions
	//	void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
	//	// Encode joint rotations into input vector 
	//	void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
	//	// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
	//	void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	//};

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

	//class RelativeEulerAnglePcaDecoder : public RelativeEulerAngleDecoder
	//{
	//public:
	//	Eigen::RowVectorXd	meanY;
	//	Eigen::MatrixXd		pcaY;
	//	Eigen::MatrixXd		invPcaY;
	//public:
	//	~RelativeEulerAnglePcaDecoder();
	//	// Decode input vector into local rotation quaternions
	//	void Decode(_Out_ array_view<DirectX::Quaternion> rots, _In_ const VectorType& x) override;
	//	// Encode joint rotations into input vector 
	//	void Encode(_In_ array_view<const DirectX::Quaternion> rots, _Out_ VectorType& x) override;
	//	// Convert Jaccobi respect Euler anglue to Jaccobi respect input vector
	//	void EncodeJacobi(_In_ array_view<const DirectX::Quaternion> rotations, _Inout_ JacobiType& jacb) override;
	//};

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