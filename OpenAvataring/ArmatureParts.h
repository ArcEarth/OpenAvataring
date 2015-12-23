#pragma once
#include "Armature.h"
#include <Eigen\Core>
#include <memory>
#include "Common\tree.h"

//#include "GaussianProcess.h"
//#include "PcaCcaMap.h"
//#include "RegressionModel.h"
//#include "StylizedIK.h"

namespace Causality
{
	class ArmaturePart;
	class ShrinkedArmature;
	using stdx::tree_node;
	using std::vector;
	using std::list;

	class IArmaturePartFeature abstract
	{
	public:
		virtual ~IArmaturePartFeature();

		// return -1 if not a uniform dimension feature
		virtual int GetDimension() const = 0;
		virtual int GetDimension(_In_ const ArmaturePart& block) const = 0;

		//void Get(_In_ const ShrinkedArmature& parts, _Out_ Eigen::RowVectorXf& feature, _In_ ArmatureFrameConstView frame);

		//virtual void Get(_In_ const ArmaturePart& block, _Out_ Eigen::RowVectorXf& feature, _In_ ArmatureFrameConstView frame);

		virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) = 0;

		virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) = 0;

		virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame, _In_ ArmatureFrameConstView last_frame, float frame_time);
	};

	// A Kinematic Block is a one-chain in the kinmatic tree, with additional anyalaze information 
	// usually constructed from shrinking the kinmatic tree
	// Each Block holds the children joints and other structural information
	// It is also common to build Multi-Level-of-Detail Block Layers
	class ArmaturePart : public tree_node<ArmaturePart>
	{
	public:
		int					Index;				// Index for this block, valid only through this layer
		vector<const Joint*>Joints;				// Contained Joints
		//ArmatureFrame			ChainFrame;			// Stores the length data and etc

		// Structural feature
		int					LoD;				// Level of detail
		int					LoG;				// Level of Grounding
		SymetricTypeEnum	SymetricType;		// symmetry type
		float				ExpandThreshold;	// The threshold to expand this part
		int					GroundIdx;

		ArmaturePart*		GroundParent;		// Path to grounded bone
		ArmaturePart*		SymetricPair;		// Path to grounded bone

		ArmaturePart*		LoDParent;		// Parent in LoD term
		vector<ArmaturePart*> LoDChildren;	// Children in LoD term

		int					AccumulatedJointCount; // The count of joints that owned by blocks who have minor index

		// Local Animation Domination properties
		ArmaturePart*		Dominator;
		list<ArmaturePart*>	Slaves;

		// Saliansy properties
		vector<string>		ActiveActions;
		vector<string>		SubActiveActions;

		// Local motion and Principle displacement datas
		//Eigen::MatrixXf						X;		// (Ac*F) x d, Muilti-clip local-motion feature matrix, Ac = Active Action Count, d = block feature dimension
		//Eigen::MatrixXf						LimitX;
		Eigen::VectorXf						Wxj;	// Jx1 Hirechical weights
		Eigen::VectorXf						Wx;		// dJx1 Weights replicated along dim
		//Eigen::MatrixXf						Pd;		// Muilti-clip Principle Displacement

		// feature Pca
		//Eigen::Pca<Eigen::MatrixXf>			ChainPca;
		//Eigen::DenseIndex					ChainPcadDim;
		//Eigen::RowVectorXf					ChainPcaMean;
		//Eigen::MatrixXf						ChainPcaMatrix;

		//PcaCcaMap							PdCca; // Muilti-clip deducted driver from Principle displacement to local motion
		//Eigen::MatrixXf						PvCorr; // ActiveActionCount x ActiveActionCount, record the pv : pv correlation 
		//float								PvDriveScore;

		//gaussian_process_regression			PdGpr;
		//double								ObrsvVar; // The cannonical value's varience give observation

		//StylizedChainIK						PdStyleIk;

		ArmaturePart();
		// Motion and geometry feature
		//bool				IsActive;			// Is this feature "Active" in energy?
		//bool				IsStable;			// Is Current state a stable state
		//float				MotionEnergy;		// Motion Energy Level
		//float				PotientialEnergy;	// Potenial Energy Level

		BoundingOrientedBox GetBoundingBox(ArmatureFrameConstView frame) const;

		//template <class FeatureType>
		//Eigen::RowVectorXf			GetFeatureVector(ArmatureFrameConstView frame, bool blockwiseLocalize = false) const;

		//template <class FeatureType>
		//void				SetFeatureVector(_Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) const;
		//template <class FeatureType>
		//size_t				GetFeatureDim() const {
		//	return FeatureType::Dimension * Joints.size();
		//}

		bool				IsEndEffector() const;		// Is this feature a end effector?
		bool				IsGrounded() const;			// Is this feature grounded? == foot semantic
		bool				IsSymetric() const;			// Is this feature a symetric pair?
		bool				IsLeft() const;				// Is this feature left part of a symtric feature
		bool				IsDominated() const;
		bool				HasSlaves() const;
	};

	// Shrink a Kinmatic Chain structure to a ArmaturePart
	ArmaturePart* ShrinkChainToBlock(const Joint* pJoint);


	class ShrinkedArmature
	{
	private:
		std::unique_ptr<ArmaturePart>	m_pRoot;
		const IArmature*				m_pArmature;
		std::vector<ArmaturePart*>		m_Parts;
	public:
		typedef std::vector<ArmaturePart*> cache_type;

		void SetArmature(const IArmature& armature);

		void ComputeWeights();

		explicit ShrinkedArmature(const IArmature& armature);

		ShrinkedArmature() = default;

		bool   empty() const { return m_pArmature == nullptr; }
		size_t size() const { return m_Parts.size(); }

		auto begin() { return m_Parts.begin(); }
		auto end() { return m_Parts.end(); }
		auto begin() const { return m_Parts.begin(); }
		auto end() const { return m_Parts.end(); }
		auto rbegin() { return m_Parts.rbegin(); }
		auto rend() { return m_Parts.rend(); }
		auto rbegin() const { return m_Parts.rbegin(); }
		auto rend() const { return m_Parts.rend(); }

		const IArmature& Armature() const { return *m_pArmature; }
		ArmaturePart* Root() { return m_pRoot.get(); }
		const ArmaturePart* Root() const { return m_pRoot.get(); }
		ArmaturePart* operator[](int id) { return m_Parts[id]; }
		const ArmaturePart* operator[](int id) const { return m_Parts[id]; }

		// the matrix which alters joints into 
		Eigen::PermutationMatrix<Eigen::Dynamic> GetJointPermutationMatrix(size_t feature_dim) const;
	};

	//template <class FeatureType>
	//inline Eigen::RowVectorXf ArmaturePart::GetFeatureVector(ArmatureFrameConstView frame, bool blockwiseLocalize) const
	//{
	//	Eigen::RowVectorXf Y(GetFeatureDim<FeatureType>());
	//	for (size_t j = 0; j < Joints.size(); j++)
	//	{
	//		auto jid = Joints[j]->ID;
	//		auto Yj = Y.middleCols<FeatureType::Dimension>(j * FeatureType::Dimension).transpose();
	//		FeatureType::Get(Yj, frame[Joints[j]->ID]);
	//	}

	//	if (parent() != nullptr && blockwiseLocalize && FeatureType::BlockwiseLocalize)
	//	{
	//		Eigen::RowVectorXf reference(FeatureType::Dimension);
	//		FeatureType::Get(reference, frame[parent()->Joints.back()->ID]);
	//		Y -= reference.replicate(1, Joints.size());
	//	}
	//	return Y;
	//}

	//template <class FeatureType>
	//inline void ArmaturePart::SetFeatureVector(ArmatureFrameView frame, const Eigen::RowVectorXf & X) const
	//{
	//	for (size_t j = 0; j < Joints.size(); j++)
	//	{
	//		auto jid = Joints[j]->ID;
	//		auto Xj = X.middleCols<FeatureType::Dimension>(j * FeatureType::Dimension);
	//		FeatureType::Set(frame[jid], Xj);
	//	}
	//}

	//inline void IArmaturePartFeature::Get(_In_ const ShrinkedArmature& parts, _Out_ Eigen::RowVectorXf& feature, _In_ ArmatureFrameConstView frame)
	//{
	//	int stIdx = 0;
	//	for (int i = 0; i < parts.size(); i++)
	//	{
	//		int dim = GetDimension(*parts[i]);
	//		feature.segment(stIdx, dim) = Get(*parts[i], frame);
	//		stIdx += dim;
	//	}
	//}

}