#include "pch_bcl.h"
#include "Armature.h"
#include "Animations.h"
#include <regex>
//#include <boost\assign.hpp>
#include <iostream>

using namespace DirectX;
using namespace Causality;

Bone::Bone()
	: LclScaling(1.0f), LclLength(1.0f), GblScaling(1.0f), GblLength(1.0f), LclTw(1.0f), GblTw(1.0f)
{
}

void Bone::UpdateGlobalData(const Bone & reference)
{
	XMVECTOR ParQ = XMLoadA(reference.GblRotation);
	XMVECTOR Q = XMQuaternionMultiply(XMLoadA(LclRotation), ParQ);
	XMVECTOR ParS = XMLoadA(reference.GblScaling);
	XMVECTOR S = ParS * XMLoadA(LclScaling);
	XMStoreA(GblRotation, Q);
	XMStoreA(GblScaling, S);

	//OriginPosition = reference.GblTranslation; // should be a constriant

	XMVECTOR V = XMLoadA(LclTranslation);

	LclLength = XMVectorGetX(XMVector3Length(V));

	V *= ParS;

	GblLength = XMVectorGetX(XMVector3Length(V));

	V = XMVector3Rotate(V, ParQ);//ParQ
	V = XMVectorAdd(V, XMLoadA(reference.GblTranslation));

	XMStoreA(GblTranslation, V);
}

// This will assuming LclTranslation is not changed
void Bone::UpdateLocalData(const Bone& reference)
{
	//OriginPosition = reference.GblTranslation;
	XMVECTOR InvParQ = reference.GblRotation;
	InvParQ = XMQuaternionInverse(InvParQ); // PqInv
	XMVECTOR Q = GblRotation;
	Q = XMQuaternionMultiply(Q, InvParQ);
	XMStoreA(LclRotation, Q);

	Q = (XMVECTOR)GblTranslation - (XMVECTOR)reference.GblTranslation;
	//Q = XMVector3Length(Q);
	//Q = XMVectorSelect(g_XMIdentityR1.v, Q, g_XMIdentityR1.v);

	GblLength = XMVectorGetX(XMVector3Length(Q));

	Q = XMVector3Rotate(Q, InvParQ);

	XMVECTOR S = XMLoadA(GblScaling);
	Q /= S;

	LclLength = XMVectorGetX(XMVector3Length(Q));

	LclTranslation = Q;

	S /= XMLoadA(reference.GblScaling);
	LclScaling = S;
}


void Bone::UpdateLocalDataByPositionOnly(const Bone & reference)
{
	XMVECTOR v0 = XMLoadA(this->GblTranslation) - XMLoadA(reference.GblTranslation);
	v0 = XMVector3InverseRotate(v0, reference.GblRotation);

	LclScaling = Vector3{ 1.0f, XMVectorGetX(XMVector3Length(v0)), 1.0f };

	// with Constraint No-X
	v0 = XMVector3Normalize(v0);
	XMFLOAT4A Sp;
	XMStoreFloat4A(&Sp, v0);
	float Roll = -std::asinf(Sp.x);
	float Pitch = std::atan2f(Sp.z, Sp.y);
	this->LclRotation = XMQuaternionRotationRollPitchYaw(Pitch, 0.0f, Roll);
	this->GblRotation = XMQuaternionMultiply(this->LclRotation, reference.GblRotation);
}

XMMATRIX Bone::TransformMatrix(const Bone & from, const Bone & to)
{
	using namespace DirectX;

	XMVECTOR VScaling = XMLoadA(to.GblScaling) / XMLoadA(from.GblScaling);
	XMVECTOR VRotationOrigin = XMVectorSelect(g_XMSelect1110.v, XMLoadA(from.GblTranslation), g_XMSelect1110.v);
	XMMATRIX MRotation = XMMatrixRotationQuaternion(XMQuaternionConjugate(XMLoadA(from.GblRotation)));
	XMVECTOR VTranslation = XMVectorSelect(g_XMSelect1110.v, XMLoadA(to.GblTranslation), g_XMSelect1110.v);

	VRotationOrigin = -VRotationOrigin;
	VRotationOrigin = XMVector3Transform(VRotationOrigin, MRotation);
	VRotationOrigin = VRotationOrigin * VScaling;
	XMMATRIX M = MRotation;
	M.r[3] = XMVectorSelect(g_XMIdentityR3.v, VRotationOrigin, g_XMSelect1110.v);

	MRotation = XMMatrixRotationQuaternion(XMLoadA(to.GblRotation));
	M = XMMatrixMultiply(M, MRotation);
	M.r[3] += VTranslation;

	//XMVECTOR toEnd = XMVector3Transform(XMLoadA(from.GblTranslation), M);

	//if (XMVector4Greater(XMVector3LengthSq(toEnd - XMLoadA(to.GblTranslation)), XMVectorReplicate(0.001)))
	//	std::cout << "NO!!!!" << std::endl;
	return M;
}

XMMATRIX Bone::RigidTransformMatrix(const Bone & from, const Bone & to)
{
	XMVECTOR rot = XMQuaternionInverse(from.GblRotation);
	rot = XMQuaternionMultiply(rot, to.GblRotation);
	XMVECTOR tra = XMLoadA(to.GblTranslation) - XMLoadA(from.GblTranslation);
	return XMMatrixRigidTransform(XMLoadA(from.GblTranslation), rot, tra);
}

XMDUALVECTOR Bone::RigidTransformDualQuaternion(const Bone & from, const Bone & to)
{
	XMVECTOR rot = XMQuaternionInverse(from.GblRotation);
	rot = XMQuaternionMultiply(rot, to.GblRotation);
	XMVECTOR tra = XMLoadA(to.GblTranslation) - XMLoadA(from.GblTranslation);
	return XMDualQuaternionRigidTransform(XMLoadA(from.GblTranslation), rot, tra);
}

ArmatureFrame::ArmatureFrame(size_t size)
	: BaseType(size)
{
}

ArmatureFrame::ArmatureFrame(const IArmature & armature)
{
	assert(this->size() == armature.size());
	auto df = armature.bind_frame();
	BaseType::assign(df.begin(), df.end());
}

//ArmatureFrame::ArmatureFrame(ArmatureFrameView frameView)
//	: BaseType(frameView.begin(), frameView.end())
//{
//}

ArmatureFrame::ArmatureFrame(const ArmatureFrameConstView &frameView)
	: BaseType(frameView.begin(), frameView.end())
{
}

ArmatureFrame::ArmatureFrame(const ArmatureFrame &) = default;

ArmatureFrame::ArmatureFrame(ArmatureFrame && rhs)
{
	*this = std::move(rhs);
}

ArmatureFrame& ArmatureFrame::operator=(const ArmatureFrame&) = default;
ArmatureFrame& ArmatureFrame::operator=(ArmatureFrame&& rhs)
{
	BaseType::_Assign_rv(std::move(rhs));
	return *this;
}
//ArmatureFrame& ArmatureFrame::operator=(ArmatureFrameView frameView)
//{
//	BaseType::assign(frameView.begin(), frameView.end());
//	return *this;
//}
ArmatureFrame& ArmatureFrame::operator=(const ArmatureFrameConstView &frameView)
{
	BaseType::assign(frameView.begin(), frameView.end());
	return *this;
}


namespace Causality
{
	void FrameRebuildGlobal(const IArmature& armature, ArmatureFrameView frame)
	{
		for (auto& joint : armature.joints())
		{
			auto& bone = frame[joint.ID];
			if (joint.is_root())
			{
				bone.GblRotation = bone.LclRotation;
				bone.GblScaling = bone.LclScaling;
				bone.GblTranslation = bone.LclTranslation;
				bone.LclLength = bone.GblLength = 1.0f; // Length of root doesnot have any meaning
			}
			else
			{
				//bone.OriginPosition = at(joint.ParentID).GblTranslation;

				bone.UpdateGlobalData(frame[joint.ParentID]);
			}
		}
	}

	void FrameRebuildLocal(const IArmature& armature, ArmatureFrameView frame)
	{
		for (auto& joint : armature.joints())
		{
			auto& bone = frame[joint.ID];
			if (joint.is_root())
			{
				bone.LclRotation = bone.GblRotation;
				bone.LclScaling = bone.GblScaling;
				bone.LclTranslation = bone.GblTranslation;
				bone.LclLength = bone.GblLength = 1.0f; // Length of root doesnot have any meaning
			}
			else
			{
				bone.UpdateLocalData(frame[joint.ParentID]);
			}
		}
	}

	void FrameLerpEst(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float t, const IArmature& armature, bool rebuild)
	{
		//assert((Armature == lhs.pArmature) && (lhs.pArmature == rhs.pArmature));
		XMVECTOR vt = XMVectorReplicate(t);
		for (size_t i = 0; i < armature.size(); i++)
		{
			XMVECTOR Q = DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclRotation), XMLoadA(rhs[i].LclRotation), vt);
			Q = _DXMEXT XMVector4Normalize(Q);
			XMStoreA(out[i].LclRotation, Q);
			XMStoreA(out[i].LclScaling, DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclScaling), XMLoadA(rhs[i].LclScaling), vt));
			XMStoreA(out[i].LclTranslation, DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclTranslation), XMLoadA(rhs[i].LclTranslation), vt));
		}
		if (rebuild)
			FrameRebuildGlobal(armature, out);
	}


	void FrameLerp(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float t, const IArmature& armature, bool rebuild)
	{
		//assert((Armature == lhs.pArmature) && (lhs.pArmature == rhs.pArmature));
		XMVECTOR vt = XMVectorReplicate(t);
		for (size_t i = 0; i < armature.size(); i++)
		{
			XMStoreA(out[i].LclRotation, DirectX::XMQuaternionSlerpV(XMLoadA(lhs[i].LclRotation), XMLoadA(rhs[i].LclRotation), vt));
			XMStoreA(out[i].LclScaling, DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclScaling), XMLoadA(rhs[i].LclScaling), vt));
			XMStoreA(out[i].LclTranslation, DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclTranslation), XMLoadA(rhs[i].LclTranslation), vt));
		}
		if (rebuild)
			FrameRebuildGlobal(armature, out);
	}

	void FrameDifference(ArmatureFrameView out, ArmatureFrameConstView from, ArmatureFrameConstView to)
	{
		auto n = std::min(from.size(), to.size());
		assert(out.size() >= n);
		for (size_t i = 0; i < n; i++)
		{
			auto& lt = out[i];
			lt.LocalTransform() = from[i].LocalTransform();
			lt.LocalTransform().Inverse();
			lt.LocalTransform() *= to[i].LocalTransform();

			lt.GlobalTransform() = from[i].GlobalTransform();
			lt.GlobalTransform().Inverse();
			lt.GlobalTransform() *= to[i].GlobalTransform();
		}
	}

	void FrameDeform(ArmatureFrameView out, ArmatureFrameConstView from, ArmatureFrameConstView deformation)
	{
		auto n = std::min(from.size(), deformation.size());
		assert(out.size() >= n);
		for (size_t i = 0; i < n; i++)
		{
			auto& lt = out[i];
			lt.LocalTransform() = from[i].LocalTransform();
			lt.LocalTransform() *= deformation[i].LocalTransform();
		}
	}

	void FrameScaleEst(_Inout_ ArmatureFrameView frame, _In_ ArmatureFrameConstView  ref, float scale)
	{
		auto n = std::min(frame.size(), ref.size());
		XMVECTOR sv = XMVectorReplicate(scale);
		for (size_t i = 0; i < n; i++)
		{
			auto& t1 = frame[i].LocalTransform();
			auto& t0 = ref[i].LocalTransform();
			auto& out = t1;
			out.Scale = XMVectorLerpV(XMLoadA(t0.Scale), XMLoadA(t1.Scale), sv);
			out.Translation = XMVectorLerpV(XMLoadA(t0.Translation), XMLoadA(t1.Translation), sv);
			out.Rotation = XMQuaternionSlerpFastV(XMLoadA(t0.Rotation), XMLoadA(t1.Rotation), sv);
		}
	}

	void FrameScale(_Inout_ ArmatureFrameView frame, _In_ ArmatureFrameConstView  ref, float scale)
	{
		auto n = std::min(frame.size(), ref.size());
		XMVECTOR sv = XMVectorReplicate(scale);
		for (size_t i = 0; i < n; i++)
		{
			auto& lt = frame[i].LocalTransform();
			auto& rt = ref[i].LocalTransform();
			IsometricTransform::LerpV(lt, rt, lt, sv);
		}
	}

	void FrameBlend(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float* blend_weights, const IArmature& armature)
	{

	}

	void FrameTransformMatrix(XMFLOAT3X4* pOut, ArmatureFrameConstView from, ArmatureFrameConstView to, size_t numOut)
	{
		using namespace std;
		size_t n = min(from.size(), to.size());
		if (numOut > 0)
			n = min(n, numOut);
		for (int i = 0; i < n; ++i)
		{
			XMMATRIX mat = Bone::TransformMatrix(from[i], to[i]);
			mat = XMMatrixTranspose(mat);
			XMStoreFloat3x4(pOut + i, mat);
		}
	}

	void FrameTransformMatrix(XMFLOAT4X4* pOut, ArmatureFrameConstView from, ArmatureFrameConstView to, size_t numOut)
	{
		using namespace std;
		size_t n = min(from.size(), to.size());
		if (numOut > 0)
			n = min(n, numOut);
		for (int i = 0; i < n; ++i)
		{
			XMMATRIX mat = Bone::TransformMatrix(from[i], to[i]);
			//mat = XMMatrixTranspose(mat);
			XMStoreFloat4x4(pOut + i, mat);
		}
	}
	Joint::Joint()
	{
		JointBasicData::ID = nullid;
		JointBasicData::ParentID = nullid;
		MirrorJoint = nullptr;
	}
	Joint::Joint(int id)
	{
		JointBasicData::ID = id;
		JointBasicData::ParentID = nullid;
		MirrorJoint = nullptr;
	}
	Joint::Joint(const JointBasicData & data)
		: JointBasicData(data)
	{
		MirrorJoint = nullptr;
	}
	Joint::Joint(const Joint & rhs)
		: JointBasicData(rhs)
	{
		MirrorJoint = nullptr;
	}
	Joint::~Joint()
	{

	}
	inline int Joint::reindex(int baseid)
	{
		int id = baseid;
		for (auto& node : this->nodes())
		{
			node.ID = id++;
			auto parent = node.parent();
			node.ParentID = parent ? parent->ID : nullid;
		}
		return id;
	}
}

StaticArmature::StaticArmature(array_view<JointBasicData> data)
{
	size_t jointCount = data.size();
	m_joints.resize(jointCount);

	for (size_t i = 0; i < jointCount; i++)
	{
		m_joints[i].SetID(i);

		int parentID = data[i].ParentID;
		if (parentID != i &&parentID >= 0)
		{
			m_joints[parentID].append_children_back(&m_joints[i]);
		}
		else
		{
			m_rootIdx = i;
		}
	}

	CaculateTopologyOrder();
}

StaticArmature::StaticArmature()
{
}

StaticArmature::StaticArmature(std::istream & file)
{
	size_t jointCount;
	file >> jointCount;

	m_joints.resize(jointCount);
	m_defaultFrame.resize(jointCount);

	// Joint line format: 
	// Hip(Name) -1(ParentID)
	// 1.5(Pitch) 2.0(Yaw) 0(Roll) 0.5(BoneLength)
	for (size_t idx = 0; idx < jointCount; idx++)
	{
		auto& joint = m_joints[idx];
		auto& bone = bind_frame()[idx];
		((JointBasicData&)joint).ID = idx;
		file >> ((JointBasicData&)joint).Name >> ((JointBasicData&)joint).ParentID;
		if (joint.ParentID != idx && joint.ParentID >= 0)
		{
			m_joints[joint.ParentID].append_children_back(&joint);
		}
		else
		{
			m_rootIdx = idx;
		}

		Vector4 vec;
		file >> vec.x >> vec.y >> vec.z >> vec.w;
		bone.LclRotation = XMQuaternionRotationRollPitchYawFromVector(vec);
		bone.LclScaling.y = vec.w;
	}

	CaculateTopologyOrder();
	FrameRebuildGlobal(*this, bind_frame());
}

StaticArmature::StaticArmature(size_t JointCount, int * Parents, const char* const* Names)
	: m_joints(JointCount)
{
	m_defaultFrame.resize(JointCount);
	for (size_t i = 0; i < JointCount; i++)
	{
		m_joints[i].SetID(i);
		m_joints[i].SetName(Names[i]);
		((JointBasicData&)m_joints[i]).ParentID = Parents[i];
		if (Parents[i] != i && Parents[i] >= 0)
		{
			m_joints[Parents[i]].append_children_back(&m_joints[i]);
		}
		else
		{
			m_rootIdx = i;
		}
	}
	CaculateTopologyOrder();
}

StaticArmature::~StaticArmature()
{
	// so that the destructor won't tries to delete multiple times
	for (auto& joint : m_joints)
	{
		joint.isolate();
	}
}

StaticArmature::StaticArmature(const IArmature & rhs)
{
	clone_from(rhs);
}

//StaticArmature::StaticArmature(const self_type & rhs)
//{
//	clone_from(rhs);
//}

StaticArmature::StaticArmature(self_type && rhs)
{
	*this = std::move(rhs);
}

StaticArmature& StaticArmature::operator=(const self_type & rhs)
{
	clone_from(rhs);
	return *this;
}

StaticArmature::self_type & StaticArmature::operator=(self_type && rhs)
{
	using std::move;
	m_rootIdx = rhs.m_rootIdx;
	m_joints = move(rhs.m_joints);
	m_order = move(rhs.m_order);
	m_defaultFrame = move(rhs.m_defaultFrame);
	return *this;
}

//void StaticArmature::clone_from(const self_type & rhs)
//{
//	this->m_joints.resize(rhs.m_joints.size());
//	for (int i = 0; i < m_joints.size(); i++)
//	{
//		this->m_joints[i] = rhs.m_joints[i];
//	}
//	this->m_order = rhs.m_order;
//	this->m_rootIdx = rhs.m_rootIdx;
//	this->m_defaultFrame = make_unique<ArmatureFrame>(*rhs.m_defaultFrame);
//
//	// Re-construct the tree structure
//	for (int i : this->m_order)
//	{
//		if (m_joints[i].ParentID >= 0 && m_joints[i].ParentID != i)
//		{
//			m_joints[m_joints[i].ParentID].append_children_back(&m_joints[i]);
//		}
//	}
//}

void StaticArmature::clone_from(const IArmature & rhs)
{
	for (auto& j : rhs.joints())
		m_order.push_back(j.ID);

	this->m_joints.resize(m_order.size());
	this->m_rootIdx = rhs.root()->ID;
	this->m_defaultFrame = rhs.bind_frame();

	// Copy Joint meta-data
	for (auto& j : rhs.joints())
		m_joints[j.ID] = j;

	// rebuild structure
	for (int i : this->m_order)
	{
		if (m_joints[i].ParentID >= 0 && m_joints[i].ParentID != i)
		{
			m_joints[m_joints[i].ParentID].append_children_back(&m_joints[i]);
		}
	}
}

//void GetBlendMatrices(_Out_ XMFLOAT4X4* pOut);

Joint * StaticArmature::at(int index) {
	return &m_joints[index];
}

Joint * StaticArmature::root()
{
	return &m_joints[m_rootIdx];
}

size_t StaticArmature::size() const
{
	return m_joints.size();
}

StaticArmature::frame_const_view StaticArmature::bind_frame() const
{
	return m_defaultFrame;
	// TODO: insert return statement here
}

void StaticArmature::set_default_frame(frame_type && frame) { m_defaultFrame = std::move(frame); }

void StaticArmature::CaculateTopologyOrder()
{
	m_order.reserve(size());
	for (auto& j : root()->nodes())
		m_order.push_back(j.ID);
}

// Lerp the local-rotation and scaling, "interpolate in Time"
//
//iterator_range<std::sregex_token_iterator> words_from_string(const std::string& str)
//{
//	using namespace std;
//	regex wordPattern("[_\\s]?[A-Za-z][a-z]*\\d*");
//	sregex_token_iterator wbegin(str.begin(), str.end(), wordPattern);
//	iterator_range<sregex_token_iterator> words(wbegin, sregex_token_iterator());
//	return words;
//}
//
//using namespace std;
//
//std::map<std::string, JointSemanticProperty>
//name2semantic = boost::assign::map_list_of
//(string("hand"), JointSemanticProperty(Semantic_Hand))
//(string("foreleg"), JointSemanticProperty(Semantic_Hand | Semantic_Foot))
//(string("arm"), JointSemanticProperty(Semantic_Hand))
//(string("claw"), JointSemanticProperty(Semantic_Hand))
//(string("wing"), JointSemanticProperty(Semantic_Hand | Semantic_Wing))
//(string("head"), JointSemanticProperty(Semantic_Head))
//(string("l"), JointSemanticProperty(Semantic_Left))
//(string("r"), JointSemanticProperty(Semantic_Right))
//(string("left"), JointSemanticProperty(Semantic_Left))
//(string("right"), JointSemanticProperty(Semantic_Right))
//(string("leg"), JointSemanticProperty(Semantic_Foot))
//(string("foot"), JointSemanticProperty(Semantic_Foot))
//(string("tail"), JointSemanticProperty(Semantic_Tail))
//(string("ear"), JointSemanticProperty(Semantic_Ear))
//(string("eye"), JointSemanticProperty(Semantic_Eye))
//(string("noise"), JointSemanticProperty(Semantic_Nouse));
//
//const JointSemanticProperty & Joint::AssignSemanticsBasedOnName()
//{
//	using namespace std;
//	using namespace boost::adaptors;
//
//	auto words = words_from_string(Name);
//	for (auto& word : words)
//	{
//		string word_str;
//		if (*word.first == '_' || *word.first == ' ')
//			word_str = std::string(word.first + 1, word.second);
//		else
//			word_str = std::string(word.first, word.second);
//
//		for (auto& c : word_str)
//		{
//			c = std::tolower(c);
//		}
//
//		this->Semantic |= name2semantic[word_str];
//	}
//	return this->Semantic;
//}

// check if two tree node is "structural-similar"
template <class Derived, bool ownnersip>
bool is_similar(_In_ const stdx::tree_node<Derived, ownnersip> *p, _In_ const stdx::tree_node<Derived, ownnersip> *q)
{
	typedef const stdx::tree_node<Derived, ownnersip> * pointer;
	pointer pc = p->first_child();
	pointer qc = q->first_child();

	if ((pc == nullptr) != (qc == nullptr))
		return false;
	else if (!(pc || qc))
		return true;

	while (pc != nullptr && qc != nullptr)
	{
		bool similar = is_similar(pc, qc);
		if (!similar) return false;
		pc = pc->next_sibling();
		qc = qc->next_sibling();
	}

	return pc == nullptr && qc == nullptr;
}

void Causality::BuildJointMirrorRelation(IArmature& armature)
{
	Joint* root = armature.root();
	ArmatureFrameConstView frame = armature.bind_frame();

	float epsilon = 1.00f;
	auto _children = root->descendants();
	std::vector<std::reference_wrapper<Joint>> children(_children.begin(), _children.end());
	std::vector<Joint*> &joints = reinterpret_cast<std::vector<Joint*> &>(children);

	for (int i = 0; i < children.size(); i++)
	{
		auto& bonei = frame[joints[i]->ID];
		auto& ti = bonei.GblTranslation;
		for (int j = i + 1; j < children.size(); j++)
		{
			auto& bonej = frame[joints[j]->ID];
			auto& tj = bonej.GblTranslation;

			if (joints[i]->parent() == joints[j]->parent() && is_similar(joints[i], joints[j]))
			{
				joints[i]->MirrorJoint = joints[j];
				joints[j]->MirrorJoint = joints[i];
			}
		}
	}
}

DynamicArmature::DynamicArmature(DynamicArmature && rhs) = default;

DynamicArmature::~DynamicArmature()
{
}

DynamicArmature::DynamicArmature(std::unique_ptr<joint_type>&& root, frame_type && defaultframe)
{
	m_root = move(root);
	m_defaultFrame = move(defaultframe);

	// build the reverse index
	for (auto& joint : m_root->nodes())
	{
		auto rib = m_index.try_emplace(joint.ID, &joint);
		assert(rib.second);
	}
}
DynamicArmature::joint_type * DynamicArmature::at(int index)
{
	auto itr = m_index.find(index);
	if (itr != m_index.end())
		return itr->second;
	else
		return nullptr;
}
DynamicArmature::joint_type * DynamicArmature::root()
{
	return m_root.get();
}
size_t DynamicArmature::size() const
{
	return m_index.size();
}
DynamicArmature::frame_const_view DynamicArmature::bind_frame() const { return m_defaultFrame; }
DynamicArmature::frame_type & DynamicArmature::bind_frame() { return m_defaultFrame; }
void DynamicArmature::set_default_frame(frame_type && frame) { m_defaultFrame = std::move(frame); }
//	void DynamicArmature::clone_from(const IArmature & rhs)
//	{
//	}
void DynamicArmature::clone_from(const joint_type & root)
{
	m_root.reset(root.clone());
	reindex();
}
//	void DynamicArmature::reroot(joint_type * pNewRoot)
//	{
//	}
//	void DynamicArmature::remove(unsigned int jointID)
//	{
//		auto itr = m_index.find(jointID);
//		assert(itr != m_index.end() && itr->second);
//		auto subtree = itr->second;
//		subtree->isolate();
//		for (auto& joint : subtree->nodes())
//			m_index.erase(joint.ID);
//		delete subtree;
//	}
//
//	void DynamicArmature::remove(joint_type * pJoint)
//	{
//		pJoint->isolate();
//		for (auto& joint : pJoint->nodes())
//		{
//			m_index.erase(joint.ID);
//		}
//		delete pJoint;
//	}
//	void DynamicArmature::append(joint_type * pTargetJoint, joint_type * pSrcJoint)
//	{
//		pTargetJoint->append_children_back(pSrcJoint);
//		reindex();
//	}
//	std::map<int, int> DynamicArmature::append(joint_type * pTargetJoint, DynamicArmature * pSkeleton, bool IsCoordinateRelative)
//	{
//		std::map<int, int> mapper;
//		if (pSkeleton == nullptr)
//			return mapper;
//
//		if (pTargetJoint != nullptr && !this->contains(pTargetJoint))
//		{
//			std::cout << "Error : pTargetJoint don't belong to the skeleton." << std::endl;
//			return mapper;
//		}
//
//		unsigned int Key = 0;
//
//		// Copy the sub-skeleton
//		Joint* pRoot = new Joint(*pSrcSubSkeleton);
//
//		if (this->empty())
//		{
//			this->m_root = pRoot;
//		}
//
//		// If don't give the append target , somehow, find one instead
//		bool DefaultInvalidFlag = false;
//		if (!pTargetJoint)
//		{
//			cout << "Warning : Don't get a specific binding bone , find it instead." << endl;
//			// We can't give a relative coordinate if we don't know it , right?
//			IsRelativeCoordinate = false;
//			pTargetJoint = this->FindClosestJoint(pRoot->Entity[Current].Position, Current);
//			DefaultInvalidFlag = true;
//		}
//
//		pTargetJoint->Children_Add(pRoot);
//
//		pRoot->for_all_in_sub_skeleton([&](Joint* pJoint) {
//			// Fix the ID
//			while (containtsKey(Key))
//				Key++;
//			mapper->insert(pair<unsigned int, unsigned int>(pJoint->ID, Key));
//			pJoint->ID = Key;
//			this->Index.insert(Kinematics::IndexedSkeleton::Index_Item(Key, pJoint));
//
//			// Deduce the hierarchical data if needed
//			if (!IsRelativeCoordinate) {
//				pJoint->Deduce_Hierarchical_from_Global(Current);
//				// If default skeleton is invalid , so , use current instead.
//				if (!DefaultInvalidFlag) {
//					pJoint->Deduce_Hierarchical_from_Global(Default);
//				}
//				else {
//					pJoint->Snap_Default_to_Current();
//				}
//
//
//			}
//		});
//
//		pTargetJoint->ReBuildSubSkeleton_Global_from_Hierarchical(Current);
//		pTargetJoint->ReBuildSubSkeleton_Global_from_Hierarchical(Default);
//		return mapper;
//		return std::map<int, int>();
//	}
std::map<int, int> DynamicArmature::reindex()
{
	std::map<int, int> mapper;
	int idx = 0;
	for (auto& joint : m_root->nodes())
	{
		mapper.try_emplace(joint.ID, idx);
		joint.ID = idx++;
		if (joint.parent())
			joint.ParentID = joint.parent()->ID;
	}
	return mapper;
}
//}
