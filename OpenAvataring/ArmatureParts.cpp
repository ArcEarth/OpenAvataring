#include "pch.h"
#include "ArmatureParts.h"
#include "Causality\Animations.h"

using namespace Causality;
using namespace DirectX;
using namespace Eigen;
using namespace std;

//bool is_symetric(const ArmaturePart& lhs, const ArmaturePart& rhs)
//{
//	return
//		(lhs.SymetricPair != nullptr && lhs.SymetricPair->Index == rhs.Index)
//		|| (rhs.SymetricPair != nullptr && rhs.SymetricPair->Index == lhs.Index);
//}

void ShrinkedArmature::SetArmature(const IArmature & armature)
{
	m_pArmature = &armature;
	m_pRoot.reset(ShrinkChainToBlock(armature.root()));
	int idx = 0, accumIdx = 0;

	for (auto& block : m_pRoot->nodes())
	{
		m_Parts.push_back(&block);
		block.Index = idx++;
		block.AccumulatedJointCount = accumIdx;
		accumIdx += block.Joints.size();
		//block.Wx = block.Wx.array().exp();
	}

	for (auto part : m_Parts)
	{
		auto pMj = part->Joints[0]->MirrorJoint;
		if (pMj != nullptr && pMj->MirrorJoint == part->Joints[0])
		{
			auto itr = std::find_if(m_Parts.begin(), m_Parts.end(), [pMj](auto part)->bool {
				return part->Joints[0] == pMj;
			});
			if (itr != m_Parts.end())
			{
				(*itr)->SymetricPair == part;
				part->SymetricPair = *itr;
			}
		}
	}
}

void ShrinkedArmature::ComputeWeights()
{
	const auto& frame = m_pArmature->default_frame();

	// Reverse Depth first visit, thus child must be visited before parent
	auto rparts = make_range(m_Parts.rbegin(), m_Parts.rend());
	for (auto& pblcok : rparts)
	{
		auto& part = *pblcok;
		float length = 0;
		for (auto& childPart : part.children())
		{
			length += childPart.Wxj[0] * childPart.Wxj[0];
		}
		length = sqrt(length);

		part.Wxj.resize(part.Joints.size());
		part.Wxj.setZero();
		for (int i = part.Joints.size() - 1; i >= 0; --i)
		{
			length += frame[part.Joints[i]->ID].LclTranslation.Length();
			part.Wxj[i] = length;
		}

		Matrix<float, -1, -1, RowMajor> wx = part.Wxj.replicate(1, 3).eval();
		part.Wx = VectorXf::Map(wx.data(), wx.size());
	}

}

ShrinkedArmature::ShrinkedArmature(const IArmature & armature)
{
	SetArmature(armature);
}

IArmaturePartFeature::~IArmaturePartFeature()
{
}

Eigen::RowVectorXf IArmaturePartFeature::Get(const ArmaturePart& block, ArmatureFrameConstView frame, ArmatureFrameConstView last_frame, float frame_time)
{
	return Get(block, frame);
}

ArmaturePart::ArmaturePart()
{
	Index = -1;
	LoD = 0;
	LoG = 0;
	SymetricType = SymetricTypeEnum::Symetric_None;
	ExpandThreshold = 0;
	GroundIdx = 0;
	GroundParent = 0;
	SymetricPair = 0;
	LoDParent = 0;
	Dominator = 0;
}

ArmaturePart* Causality::ShrinkChainToBlock(const Joint* pJoint)
{
	if (pJoint == nullptr || pJoint->is_null()) return nullptr;
	ArmaturePart* pBlock = new ArmaturePart;

	auto pChild = pJoint;
	do
	{
		pBlock->Joints.push_back(pChild);
		pJoint = pChild;
		pChild = pChild->first_child();
	} while (pChild && !pChild->next_sibling());

	for (auto& child : pJoint->children())
	{
		auto childBlcok = ShrinkChainToBlock(&child);
		pBlock->append_children_back(childBlcok);
	}

	return pBlock;
}

Eigen::PermutationMatrix<Eigen::Dynamic> ShrinkedArmature::GetJointPermutationMatrix(size_t feature_dim) const
{
	using namespace Eigen;

	// Perform a Column permutation so that feature for block wise is consist
	// And remove the position data!!!

	auto N = m_pArmature->size();
	vector<int> indices(N);
	int idx = 0;

	for (auto pBlock : m_Parts)
	{
		for (auto& pj : pBlock->Joints)
		{
			//indices[pj->ID] = idx++;
			indices[idx++] = pj->ID;
		}
	}

	MatrixXi a_indvec = RowVectorXi::Map(indices.data(), indices.size()).replicate(feature_dim, 1);
	a_indvec.array() *= feature_dim;
	a_indvec.colwise() += Matrix<int, -1, 1>::LinSpaced(feature_dim, 0, feature_dim - 1);

#ifdef _DEBUG
	cout << a_indvec << endl;
#endif

	Eigen::PermutationMatrix<Eigen::Dynamic> perm(VectorXi::Map(a_indvec.data(), N*feature_dim));
	return perm;
}
