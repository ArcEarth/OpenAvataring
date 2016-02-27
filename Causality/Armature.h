#pragma once
#include <unordered_map>
#include "Math3D.h"
#include "Common\tree.h"
#include "String.h"
#include <iosfwd>
//#include <boost\range\adaptor\transformed.hpp>

namespace Causality
{
	class IArmature;
	class ArmaturePart;
	class ShrinkedArmature;
	class ArmatureFrame;

	// Pure pose and Dynamic data for a bone
	// "Structure" information is not stored here
	// Each bone is a affine transform (Scale-Rotate-Translate) 
	// the bone will store this transform in Local frame and global frame
	XM_ALIGNATTR
	struct Bone : public AlignedNew<XMVECTOR>
	{
	public:
		// Local Data
		XM_ALIGNATTR
		Quaternion LclRotation;
		// Local Translation, represent the offset vector (Pe - Po) in current frame 
		// Typical value should always be (0,l,0), where l = length of the bone
		// Should be constant among time!!!
		XM_ALIGNATTR
		Vector3	LclTranslation;
		float				LclTw; // Padding, w-coff of local translation

		XM_ALIGNATTR
		Vector3	LclScaling; // Local Scaling , adjust this transform to adjust bone length
		float				LclLength;  // offset should = 16 + 3x4 = 28, the Length before any Scaling

		// Global Data (dulplicate with Local)
		// Global Rotation
		XM_ALIGNATTR
		Quaternion GblRotation;
		// Global Position for the ending joint of this bone
		// Aka : End-Position
		XM_ALIGNATTR
		Vector3	GblTranslation;
		float				GblTw; // Padding
		XM_ALIGNATTR
		Vector3	GblScaling;
		float				GblLength;		// Length of this bone, after scaling


		//XM_ALIGNATTR
		//Vector3	OriginPosition;	// Reference point of a bone

		//XMVECTOR EndJointPosition() const;
		//bool DirtyFlag;

		Bone();

		void GetBoundingBox(BoundingBox& out) const;
		// Update from Hirachy or Global
		// FK Caculation
		// Need this.LclRotation, this.LclScaling, and all Data for reference
		void UpdateGlobalData(const Bone& refernece);
		// Easy-IK Caculation with No-X-Rotation-Constraint and bone length will be modified if the transform is not isometric
		void UpdateLocalDataByPositionOnly(const Bone& reference);
		// Assuming Global position & orientation is known
		void UpdateLocalData(const Bone& reference);

		inline const IsometricTransform& LocalTransform() const { return reinterpret_cast<const IsometricTransform&>(*this); }
		inline IsometricTransform& LocalTransform() { return reinterpret_cast<IsometricTransform&>(*this); }
		inline const IsometricTransform& GlobalTransform() const { return reinterpret_cast<const IsometricTransform&>(this->GblRotation); }
		inline IsometricTransform& GlobalTransform() { return reinterpret_cast<IsometricTransform&>(this->GblRotation); }
	public:
		// Static helper methods for caculate transform matrix
		// Caculate the Transform Matrix from "FromState" to "ToState"
		static XMMATRIX TransformMatrix(const Bone& from, const Bone& to);
		// Ingnore the LclScaling transform, have better performence than TransformMatrix
		static XMMATRIX RigidTransformMatrix(const Bone& from, const Bone& to);
		// Ingnore the LclScaling transform, may be useful in skinning with Dual-Quaternion
		static XMDUALVECTOR RigidTransformDualQuaternion(const Bone& from, const Bone& to);

	public:
		// number of float elements per bone
		static const auto BoneWidth = 24;

		//typedef Eigen::Map<Eigen::Matrix<float, BoneWidth, 1>, Eigen::Aligned> EigenType;
		//EigenType AsEigenType()
		//{
		//	return EigenType(&LclRotation.x);
		//}
	};

	XM_ALIGNATTR
	struct BoneIntrinsic
	{
		Vector3 LclTranslation;
		float	_padding0;
		Vector3 LclScale;
		float	_padding1;
	};

	XM_ALIGNATTR
	struct BoneExtrinsic
	{
		Quaternion	LclRotation;
		Quaternion	GblRotation;
		Vector3		GblTranslation;
		float		_padding;
	};

	static_assert(offsetof(Bone, GblRotation) == sizeof(IsometricTransform),"Compilier not supported.");

	XM_ALIGNATTR
	struct BoneVelocity
	{
		XM_ALIGNATTR
		Vector3 LinearVelocity;
		XM_ALIGNATTR
		Vector3 AngelarVelocity;
	};

	XM_ALIGNATTR
	struct BoneSplineNode : public Bone, public BoneVelocity
	{
	};

	typedef std::vector<DirectX::IsometricTransform, DirectX::XMAllocator> IsometricTransformFrame;

	typedef array_view<Bone> ArmatureFrameView;
	typedef array_view<const Bone> ArmatureFrameConstView;

	class ArmatureFrame : public std::vector<Bone, DirectX::XMAllocator>
	{
	public:
		typedef ArmatureFrame self_type;

		typedef public std::vector<Bone, DirectX::XMAllocator> BaseType;
		using BaseType::operator[];
		//using BaseType::operator=;

		ArmatureFrame() = default;
		explicit ArmatureFrame(size_t size);
		// copy from default frame
		explicit ArmatureFrame(const IArmature& armature);
		explicit ArmatureFrame(const ArmatureFrameConstView & frameView);
		ArmatureFrame(const ArmatureFrame&);
		ArmatureFrame(ArmatureFrame&& rhs);
		ArmatureFrame& operator=(const ArmatureFrame&);
		ArmatureFrame& operator=(ArmatureFrame&& rhs);
		ArmatureFrame& operator=(const ArmatureFrameConstView &frameView);


		// number of float elements per bone
		static const auto BoneWidth = sizeof(Bone) / sizeof(float);

	};

	void FrameRebuildGlobal(const IArmature& armature, ArmatureFrameView frame);
	void FrameRebuildLocal(const IArmature& armature, ArmatureFrameView frame);

	// Lerp the local-rotation and scaling, "interpolate in Time"
	void FrameLerp(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float t, const IArmature& armature, bool rebuild = true);
	void FrameLerpEst(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float t, const IArmature& armature, bool rebuild = true);
	void FrameScaleEst(_Inout_ ArmatureFrameView frame, _In_ ArmatureFrameConstView  ref, float scale);

	void FrameDifference(ArmatureFrameView out, ArmatureFrameConstView from, ArmatureFrameConstView to);
	void FrameDeform(ArmatureFrameView out, ArmatureFrameConstView from, ArmatureFrameConstView deformation);

	// Blend Two Animation Frame, "Blend different parts in Space"
	void FrameBlend(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float* blend_weights, const IArmature& armature);
	void FrameScale(ArmatureFrameView frame, ArmatureFrameConstView ref, float scale);

	void FrameTransformMatrix(DirectX::XMFLOAT3X4* pOut, ArmatureFrameConstView from, ArmatureFrameConstView to, size_t numOut = 0);
	void FrameTransformMatrix(DirectX::XMFLOAT4X4* pOut, ArmatureFrameConstView from, ArmatureFrameConstView to, size_t numOut = 0);

	class BoneVelocityFrame : public std::vector<BoneVelocity, DirectX::XMAllocator>
	{
	public:
		typedef std::vector<BoneVelocity, DirectX::XMAllocator> base_type;
		using base_type::base_type;
		using base_type::operator[];
		using base_type::operator=;
	};

	struct RotationConstriant
	{
		Vector3 UpperBound;
		Vector3 LowerBound;
	};

	enum JointSemantic
	{
		Semantic_None = 0x0,
		Semantic_Hand = 0x1,
		Semantic_Wing = 0x2,
		Semantic_Foot = 0x4,
		Semantic_Tail = 0x8,
		Semantic_Mouse = 0x10,
		Semantic_Head = 0x20,
		Semantic_Eye = 0x40,
		Semantic_Nouse = 0x80,
		Semantic_Ear = 0x100,

		Semantic_Left = 0x1000000,
		Semantic_Right = 0x2000000,
		Semantic_Center = 0x4000000,
	};

	using JointSemanticProperty = unsigned;//CompositeFlag<JointSemantic>;
	// A Joint descript the structure information of a joint
	// Also represent the "bone" "end" with it
	// the state information could be retrived by using it's ID
	struct JointBasicData
	{
		// The Index for this joint (&it's the bone ending with it)
		// The Name for this joint
		int							ID;
		// Should be -1 for Root
		int							ParentID;
		string						Name;
	};

	struct ColorHistogram
	{
	};

	class Joint : public stdx::tree_node<Joint>, public JointBasicData
	{
	public:
		static constexpr int		nullid = -1;
		Joint*						MirrorJoint; // the symetric pair of this joint
		JointSemanticProperty		Semantic;
		RotationConstriant			RotationConstraint;
		Vector3						Scale;
		Vector3						OffsetFromParent;

		// Saliency parameters & Extra Parameters
		Color						AverageColor;
		ColorHistogram				ColorDistribution;
		Vector3						PositionDistribution;
		BoundingBox					VelocityDistribution;

		float						IntrinsicSaliency;
		float						ExtrinsicSaliency;

	public:
		Joint();

		explicit Joint(int id);

		explicit Joint(const JointBasicData& data);
		Joint(const Joint& rhs);
		~Joint();

		void SetID(int idx) { JointBasicData::ID = idx; }

		void SetName(const string& name) { JointBasicData::Name = name; }
		void SetParentID(int id) { JointBasicData::ParentID = id; }

		const JointSemanticProperty& AssignSemanticsBasedOnName();

		int reindex(int baseid);

		//const JointSemanticProperty& Semantics() const;
		//JointSemanticProperty& Semantics();

		//const RotationConstriant&	RotationConstraint() const;
		//void SetRotationConstraint(const RotationConstriant&);
	};

	// Represent a model "segmentation" and it's hireachy
	class IArmature
	{
	public:
		typedef Joint joint_type;

		typedef ArmatureFrame frame_type;
		typedef ArmatureFrameView frame_view;
		typedef ArmatureFrameConstView frame_const_view;

		virtual ~IArmature() {}

		virtual Joint* at(int index) = 0;
		virtual Joint* root() = 0;
		inline const Joint* root() const
		{
			return const_cast<IArmature*>(this)->root();
		}
		virtual size_t size() const = 0;
		virtual frame_const_view bind_frame() const = 0;

		bool empty() { return root() == nullptr; }
		auto joints() const 
		{
			return root()->nodes();
		}

		auto joints()
		{
			return root()->nodes();
		}

		// We say one Skeleton is compatiable with another Skeleton rhs
		// if and only if rhs is a "base tree" of "this" in the sense of "edge shrink"
		//bool IsCompatiableWith(IArmature* rhs) const;

		inline const Joint* at(int index) const
		{
			return const_cast<IArmature*>(this)->at(index);
		}



		inline Joint* operator[](int idx)
		{
			return at(idx);
		}

		inline const Joint* operator[](int idx) const
		{
			return at(idx);
		}

		inline const Bone& default_bone(int index) const
		{
			return bind_frame()[at(index)->ID];
		}

		//std::vector<size_t> m_order;
	};

	void BuildJointMirrorRelation(IArmature& armature);

	enum SymetricTypeEnum
	{
		Symetric_None = 0,
		Symetric_Left,
		Symetric_Right,
	};

	// The Skeleton which won't change it's structure in runtime
	class StaticArmature : public IArmature
	{
	public:
		typedef StaticArmature self_type;

	private:
		int							m_rootIdx;
		vector<joint_type>			m_joints;
		vector<int>					m_order; //Topological order of joints
		frame_type					m_defaultFrame;

	public:

		StaticArmature();
		StaticArmature(array_view<JointBasicData> data);
		// deserialization
		StaticArmature(std::istream& file);
		StaticArmature(size_t JointCount, int *JointsParentIndices, const char* const* Names);
		StaticArmature(const IArmature& rhs);
		StaticArmature(self_type&& rhs);
		~StaticArmature();

		self_type& operator=(const self_type& rhs);
		self_type& operator=(self_type&& rhs);

		//void clone_from(const self_type& rhs);
		void clone_from(const IArmature & rhs);

		//void GetBlendMatrices(_Out_ XMFLOAT4X4* pOut);
		virtual joint_type* at(int index) override;
		virtual joint_type* root() override;
		virtual size_t size() const override;
		virtual frame_const_view bind_frame() const override;
		frame_type& bind_frame() { return m_defaultFrame; }
		void set_default_frame(frame_type &&pFrame);
		// A topolical ordered joint index sequence
		const std::vector<int>& joint_indices() const
		{
			return m_order;
		}

		//auto joints() const //-> decltype(adaptors::transform(m_order,function<const Joint&(int)>()))
		//{
		//	function<const Joint&(int)> func = [this](int idx)->const joint_type& {return m_joints[idx]; };
		//	return transform(m_order, func);
		//}

		//auto joints() //-> decltype(adaptors::transform(m_order, function<Joint&(int)>()))
		//{
		//	using namespace boost::adaptors;
		//	function<Joint&(int)> func = [this](int idx)->joint_type&{ return m_joints[idx]; };
		//	return transform(m_order, func);
		//}


		using IArmature::operator[];
	protected:
		void CaculateTopologyOrder();
	};

	// A armature which fellows a graph structure
	class GraphArmature
	{
	};

	// The armature strucuture which provide full ability in editing
	// but lesser performance in query
	class DynamicArmature : public IArmature
	{
	public:
		typedef DynamicArmature self_type;
		typedef std::pair<int, joint_type*> Index_Item;

		void clone_from(const IArmature& rhs);
		void clone_from(const joint_type& root);

		void reroot(joint_type* pNewRoot);
		void remove(unsigned int jointID);
		void remove(joint_type* pJoint);

		DynamicArmature();
		~DynamicArmature();

		explicit DynamicArmature(const IArmature& rhs);
		explicit DynamicArmature(DynamicArmature&& rhs);
		explicit DynamicArmature(std::unique_ptr<joint_type>&& root, frame_type&& defaultframe);
		explicit DynamicArmature(const joint_type & root);

		std::map<int, int> append(joint_type* pTargetJoint, std::unique_ptr<joint_type>&& pSrcJoint);
		std::map<int, int> append(joint_type* pTargetJoint, DynamicArmature* pSkeleton, bool IsCoordinateRelative = false);

		// This method adjust Joint Index automaticly to DFS order
		// Anyway , lets just return the value since we have Rvalue && move sementic now
		std::map<int, int> reindex();

		bool contains(const joint_type* pJoint) const {
			if (!pJoint) return false;
			auto itr = this->m_index.find(pJoint->ID);
			return (itr != m_index.end()) && (itr->second == pJoint);
		}

		bool contains(unsigned int jointID) const {
			return (m_index.find(jointID) != m_index.end());
		}

		virtual joint_type* at(int index) override;

		virtual joint_type* root() override;

		virtual size_t size() const override;

		virtual frame_const_view bind_frame() const override;

		frame_type& bind_frame();
		void set_default_frame(frame_type &&frame);

	protected:
		unique_ptr<joint_type>			m_root;
		std::map<int, Joint*>			m_index;
		frame_type						m_defaultFrame;
	};

}