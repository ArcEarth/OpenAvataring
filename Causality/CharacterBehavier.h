#pragma once
#include "Armature.h"
#include "Animations.h"

namespace Causality
{

	using std::vector;
	// Represent an semantic collection of animations
	// It's the possible pre-defined actions for an given object
	class BehavierSpace
	{
	public:
		typedef ArmatureFrameAnimation animation_type;
		typedef ArmatureFrame frame_type;
		typedef BoneVelocityFrame velocity_frame_type;
		typedef vector<ArmatureFrameAnimation> container_type;

	private:
		IArmature*						m_pArmature;
		vector<animation_type>			m_AnimClips;

	public:
#pragma region Animation Clip Interfaces
		container_type& Clips() { return m_AnimClips; }
		const container_type& Clips() const { return m_AnimClips; }
		animation_type& operator[](const std::string& name);
		const animation_type& operator[](const std::string& name) const;
		void AddAnimationClip(animation_type&& animation);
		animation_type& AddAnimationClip(const std::string& name);
		bool Contains(const std::string& name) const;
#pragma endregion

		//void					UpdateArmatureParts();

		const IArmature&		Armature() const;
		IArmature&				Armature();
		void					SetArmature(IArmature& armature);
		const frame_type&		RestFrame() const; // RestFrame should be the first fram in Rest Animation

		void					UniformQuaternionsBetweenClips();
	};

}