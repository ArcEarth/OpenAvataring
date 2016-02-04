#include "pch_bcl.h"
#include "CharacterBehavier.h"

using namespace Causality;
using namespace DirectX;
using namespace std;

BehavierSpace::animation_type & BehavierSpace::operator[](const std::string & name)
{
	for (auto& clip : m_AnimClips)
	{
		if (clip.Name == name) return clip;
	}
	throw std::out_of_range("No animation with specified name exist in this aniamtion space.");
}

const BehavierSpace::animation_type & BehavierSpace::operator[](const std::string & name) const
{
	for (const auto& clip : m_AnimClips)
	{
		if (clip.Name == name) return clip;
	}
	throw std::out_of_range("No animation with specified name exist in this aniamtion space.");
}

void Causality::BehavierSpace::AddAnimationClip(animation_type && animation)
{
#ifdef _DEBUG
	for (auto& clip : m_AnimClips)
		if (clip.Name == animation.Name)
		throw std::out_of_range("Name of animation is already existed.");
#endif
	m_AnimClips.emplace_back(std::move(animation));
}

BehavierSpace::animation_type & BehavierSpace::AddAnimationClip(const std::string & name)
{
#ifdef _DEBUG
	for (auto& clip : m_AnimClips)
		if (clip.Name == name)
		throw std::out_of_range("Name of animation is already existed.");
#endif
	m_AnimClips.emplace_back(name);
	auto& anim = m_AnimClips.back();
	anim.SetArmature(*m_pArmature);
	return anim;
}

bool Causality::BehavierSpace::Contains(const std::string & name) const
{
	for (auto& clip : m_AnimClips)
	{
		if (clip.Name == name) return true;
	}
	return false;
}

//void Causality::BehavierSpace::UpdateArmatureParts()
//{
//	//auto& armature = *m_pArmature;
//}

const IArmature & BehavierSpace::Armature() const { return *m_pArmature; }

IArmature & BehavierSpace::Armature() { return *m_pArmature; }

void BehavierSpace::SetArmature(IArmature & armature) {
	assert(this->Clips().empty());
	m_pArmature = &armature;
	//UpdateArmatureParts();
}

BehavierSpace::frame_const_view BehavierSpace::RestFrame() const { return Armature().default_frame(); }

void BehavierSpace::UniformQuaternionsBetweenClips()
{
	auto& armature = *m_pArmature;
	auto& clips = m_AnimClips;

	// Try to unify rotation pivot direction
	typedef vector<Vector4, DirectX::AlignedAllocator<Vector4, alignof(DirectX::XMVECTOR)>> aligned_vector_of_vector4;
	aligned_vector_of_vector4 gbl_pivots(armature.size());
	bool gbl_def = true;
	for (auto& anim : clips)
	{
		using namespace DirectX;
		aligned_vector_of_vector4 pivots(armature.size());
		for (auto& frame : anim.GetFrameBuffer())
		{
			for (size_t i = 0; i < armature.size(); i++)
			{
				pivots[i] += XMLoadA(frame[i].LclRotation);
			}
		}

		if (gbl_def)
		{
			for (size_t i = 0; i < armature.size(); i++)
			{
				gbl_pivots[i] = DirectX::XMVector3Normalize(XMLoadA(pivots[i]));
			}
			gbl_def = false;
		}
		else
		{
			//for (size_t i = 0; i < armature.size(); i++)
			//{
			//	XMVECTOR pivot = DirectX::XMVector3Normalize(pivots[i].LoadA());
			//	XMVECTOR gbl_pivot = gbl_pivots[i].Load();
			//	if (XMVector4Less(XMVector3Dot(gbl_pivot, pivot), XMVectorZero()))
			//	{
			//		for (auto& frame : anim.GetFrameBuffer())
			//		{
			//			frame[i].LclRotation.StoreA(-frame[i].LclRotation.LoadA());
			//		}
			//	}
			//}
		}
	}
}
