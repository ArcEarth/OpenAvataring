#pragma once
#include "Causality\Animations.h"
#include <atomic>
#include "ClipMetric.h"
#include "StylizedIK.h"
#include "Causality\Serialization.h"

namespace tinyxml2
{
	class XMLElement;
}

namespace Causality
{
	class ClipInfo;
	class CharacterObject;
	class InputClipInfo;
	class CharacterClipinfo;
	class ClipFacade;

	void RemoveFrameRootTransform(ArmatureFrameView frame, const IArmature& armature);

	class CharacterController
	{
	public:
		~CharacterController();
		CharacterController();
		void Initialize(CharacterObject& character, const ParamArchive* settings);

		const ArmatureTransform& Binding() const;
		ArmatureTransform& Binding();
		std::mutex& GetBindingMutex();
		void SetBinding(std::unique_ptr<ArmatureTransform> &&upBinding);

		const ArmatureTransform& SelfBinding() const;
		ArmatureTransform& SelfBinding();

		const CharacterObject& Character() const;
		CharacterObject& Character();

		const IArmature& Armature() const;
		IArmature& Armature();

		const ShrinkedArmature& ArmatureParts() const;
		ShrinkedArmature& ArmatureParts();

		const std::vector<int>&	ActiveParts() const;
		const std::vector<int>&	SubactiveParts() const;


		float UpdateTargetCharacter(ArmatureFrameConstView sourceFrame, ArmatureFrameConstView lastSourceFrame, double deltaTime_seconds) const;

		void  SetReferenceSourcePose(const Bone& sourcePose);

		float GetLastUpdateLikilyhood() const;

		void SychronizeRootDisplacement(const Causality::Bone & bone) const;

		float CreateControlBinding(const ClipFacade& inputClip);

		array_view<std::pair<Vector3, Vector3>> PvHandles() const;
		std::vector<std::pair<Vector3, Vector3>>& PvHandles();


		std::atomic_bool			IsReady;
		int							ID;
		ArmatureFrame				PotientialFrame;

		float						CharacterScore;
		Vector3						MapRefPos;
		Vector3						CMapRefPos;
		mutable Vector3				LastPos;
		Quaternion					MapRefRot;
		Quaternion					CMapRefRot;
		int							CurrentActionIndex;

		Eigen::MatrixXf				XabpvT; // Pca matrix of Xabpv
		Eigen::RowVectorXf			uXabpv; // Pca mean of Xabpv


		// Addtional velocity
		Vector3						Vaff;

		//mutable
		size_t						m_trajectoryLength;
		std::vector<std::pair<Vector3, Vector3>> m_PvHandles;
		std::vector<std::deque<Vector3>> m_handelTrajectory;

		void push_handle(int pid, const std::pair<Vector3, Vector3>& handle);

		inline std::pair<Vector3, Vector3> GetPvHandle(int pid) const { return m_PvHandles[pid]; }

		inline auto GetPvHandleTrajectory(int pid)
		{
			return std::make_range(m_handelTrajectory[pid].rbegin(), m_handelTrajectory[pid].rend());
		}

		// Principle displacement driver
		CharacterClipinfo& GetClipInfo(const std::string& name);
		const CharacterClipinfo& GetClipInfo(const std::string& name) const
		{
			return const_cast<CharacterController&>(*this).GetClipInfo(name);
		}

		std::vector<CharacterClipinfo>& GetClipInfos() { return m_Clipinfos; }
		array_view<const CharacterClipinfo> GetClipInfos() const { return m_Clipinfos; }

		StylizedChainIK& GetStylizedIK(int pid) { return *m_SIKs[pid]; }
		const StylizedChainIK& GetStylizedIK(int pid) const { return *m_SIKs[pid]; }

		CharacterClipinfo& GetUnitedClipinfo() { return m_cpxClipinfo; }
		const CharacterClipinfo& GetUnitedClipinfo() const { return m_cpxClipinfo; }
	protected:
		// Cache frame for character
		mutable
		ArmatureFrame											m_charaFrame;
		ShrinkedArmature										m_charaParts;
		CharacterObject*										m_pCharacter;
		std::vector<CharacterClipinfo>							m_Clipinfos;

		// A Clipinfo which encapture all frames from clips
		CharacterClipinfo										m_cpxClipinfo;
		std::vector<uptr<StylizedChainIK>>						m_SIKs;

		mutable
		std::mutex												m_bindMutex;

		std::unique_ptr<ArmatureTransform>						m_pBinding;
		std::unique_ptr<ArmatureTransform>						m_pSelfBinding;

		std::vector<int>										m_ActiveParts;  // it's a set
		std::vector<int>										m_SubactiveParts;
	protected:
		void SetTargetCharacter(CharacterObject& object);

		Eigen::MatrixXf GenerateXapv(const std::vector<int> &activeParts);

		void InitializeAcvtivePart(ArmaturePart & part, tinyxml2::XMLElement * settings);
		void InitializeSubacvtivePart(ArmaturePart & part, const Eigen::MatrixXf& Xabpv, tinyxml2::XMLElement * settings);
	};

}