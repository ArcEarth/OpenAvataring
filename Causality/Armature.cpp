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
	XMStoreA(GblRotation,Q);
	XMStoreA(GblScaling,S);

	//OriginPosition = reference.GblTranslation; // should be a constriant

	XMVECTOR V = XMLoadA(LclTranslation);

	LclLength = XMVectorGetX(XMVector3Length(V));

	V *= ParS;

	GblLength = XMVectorGetX(XMVector3Length(V));

	V = XMVector3Rotate(V, ParQ);//ParQ
	V = XMVectorAdd(V, XMLoadA(reference.GblTranslation));

	XMStoreA(GblTranslation,V);
}

// This will assuming LclTranslation is not changed
void Bone::UpdateLocalData(const Bone& reference)
{
	//OriginPosition = reference.GblTranslation;
	XMVECTOR InvParQ = reference.GblRotation;
	InvParQ = XMQuaternionInverse(InvParQ); // PqInv
	XMVECTOR Q = GblRotation;
	Q = XMQuaternionMultiply(Q, InvParQ);
	XMStoreA(LclRotation,Q);

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
	: BaseType(armature.default_frame())
{
	assert(this->size() == armature.size());
}

ArmatureFrame::ArmatureFrame(ArmatureFrameView frameView)
	: BaseType(frameView.begin(), frameView.end())
{
}

ArmatureFrame::ArmatureFrame(ArmatureFrameConstView frameView)
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
ArmatureFrame& ArmatureFrame::operator=(ArmatureFrameView frameView)
{
	BaseType::assign(frameView.begin(), frameView.end());
	return *this;
}
ArmatureFrame& ArmatureFrame::operator=(ArmatureFrameConstView frameView)
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

	void FrameLerp(ArmatureFrameView out, ArmatureFrameConstView lhs, ArmatureFrameConstView rhs, float t, const IArmature& armature)
	{
		//assert((Armature == lhs.pArmature) && (lhs.pArmature == rhs.pArmature));
		XMVECTOR vt = XMVectorReplicate(t);
		for (size_t i = 0; i < armature.size(); i++)
		{
			XMStoreA(out[i].LclRotation, DirectX::XMQuaternionSlerpV(XMLoadA(lhs[i].LclRotation), XMLoadA(rhs[i].LclRotation), vt));
			XMStoreA(out[i].LclScaling, DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclScaling), XMLoadA(rhs[i].LclScaling), vt));
			XMStoreA(out[i].LclTranslation, DirectX::XMVectorLerpV(XMLoadA(lhs[i].LclTranslation), XMLoadA(rhs[i].LclTranslation), vt));
		}
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
}


StaticArmature::StaticArmature(array_view<JointBasicData> data)
{
	size_t jointCount = data.size();
	Joints.resize(jointCount);

	for (size_t i = 0; i < jointCount; i++)
	{
		Joints[i].SetID(i);

		int parentID = data[i].ParentID;
		if (parentID != i &&parentID >= 0)
		{
			Joints[parentID].append_children_back(&Joints[i]);
		}
		else
		{
			RootIdx = i;
		}
	}

	CaculateTopologyOrder();
}

StaticArmature::StaticArmature(std::istream & file)
{
	size_t jointCount;
	file >> jointCount;

	Joints.resize(jointCount);
	DefaultFrame.reset(new ArmatureFrame(jointCount));

	// Joint line format: 
	// Hip(Name) -1(ParentID)
	// 1.5(Pitch) 2.0(Yaw) 0(Roll) 0.5(BoneLength)
	for (size_t idx = 0; idx < jointCount; idx++)
	{
		auto& joint = Joints[idx];
		auto& bone = default_frame()[idx];
		((JointBasicData&)joint).ID = idx;
		file >> ((JointBasicData&)joint).Name >> ((JointBasicData&)joint).ParentID;
		if (joint.ParentID != idx && joint.ParentID >= 0)
		{
			Joints[joint.ParentID].append_children_back(&joint);
		}
		else
		{
			RootIdx = idx;
		}

		Vector4 vec;
		file >> vec.x >> vec.y >> vec.z >> vec.w;
		bone.LclRotation = XMQuaternionRotationRollPitchYawFromVector(vec);
		bone.LclScaling.y = vec.w;
	}

	CaculateTopologyOrder();
	FrameRebuildGlobal(*this, default_frame());
}

StaticArmature::StaticArmature(size_t JointCount, int * Parents, const char* const* Names)
	: Joints(JointCount)
{
	DefaultFrame.reset(new ArmatureFrame(JointCount));
	for (size_t i = 0; i < JointCount; i++)
	{
		Joints[i].SetID(i);
		Joints[i].SetName(Names[i]);
		((JointBasicData&)Joints[i]).ParentID = Parents[i];
		if (Parents[i] != i && Parents[i] >= 0)
		{
			Joints[Parents[i]].append_children_back(&Joints[i]);
		}
		else
		{
			RootIdx = i;
		}
	}
	CaculateTopologyOrder();
}

StaticArmature::~StaticArmature()
{

}

StaticArmature::StaticArmature(self_type && rhs)
{
	*this = std::move(rhs);
}

StaticArmature::self_type & StaticArmature::operator=(self_type && rhs)
{
	using std::move;
	RootIdx = rhs.RootIdx;
	Joints = move(rhs.Joints);
	TopologyOrder = move(rhs.TopologyOrder);
	DefaultFrame = move(rhs.DefaultFrame);
	return *this;
}

//void GetBlendMatrices(_Out_ XMFLOAT4X4* pOut);

Joint * StaticArmature::at(int index) {
	return &Joints[index];
}

Joint * StaticArmature::root()
{
	return &Joints[RootIdx];
}

size_t StaticArmature::size() const
{
	return Joints.size();
}

const IArmature::frame_type & StaticArmature::default_frame() const
{
	return *DefaultFrame;
	// TODO: insert return statement here
}

void Causality::StaticArmature::set_default_frame(uptr<frame_type> && frame) { DefaultFrame = std::move(frame); }

void StaticArmature::CaculateTopologyOrder()
{
	TopologyOrder.reserve(size());
	for (auto& j : root()->nodes())
		TopologyOrder.push_back(j.ID);
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

const Bone & Causality::IArmature::default_bone(int index) const
{
	return default_frame()[at(index)->ID];
}

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

void Causality::BuildJointMirrorRelation(Joint* root, ArmatureFrameConstView frame)
{
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
