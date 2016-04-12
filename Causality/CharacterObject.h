#pragma once
#include "VisualObject.h"
#include "Armature.h"
#include "CharacterBehavier.h"
#include "Common\Filter.h"
#include <atomic>
#include <mutex>

namespace Causality
{
	// Represent an Character that have intrinsic animation
	class CharacterObject : public VisualObject
	{
	public:
		typedef BehavierSpace::frame_type frame_type;
		typedef vector<BoneVelocity, DirectX::XMAllocator> velocity_frame_type;

		CharacterObject();
		~CharacterObject();

		virtual void					Parse(const ParamArchive* store) override;

		void							EnabeAutoDisplacement(bool is_enable);

		const frame_type&				GetCurrentFrame() const;
		frame_type&						MapCurrentFrameForUpdate();
		void							ReleaseCurrentFrameFrorUpdate();

		IArmature&						Armature();
		const IArmature&				Armature() const;
		BehavierSpace&					Behavier();
		const BehavierSpace&			Behavier() const;
		void							SetBehavier(BehavierSpace& behaver);

		const ArmatureFrameAnimation*	CurrentAction() const;
		string							CurrentActionName() const;
		bool							StartAction(const string& key, time_seconds begin_time = time_seconds(0), bool loop = false, time_seconds transition_time = time_seconds(0));
		bool							StopAction(time_seconds transition_time = time_seconds(0));
		const XMMATRIX*					GetBoneTransforms() const;

		bool							IsFreezed() const;
		void							SetFreeze(bool freeze);

		virtual void					SetRenderModel(DirectX::Scene::IModelNode* pMesh, int LoD = 0) override;

		virtual void					Update(time_seconds const& time_delta) override;

		// Inherited via IVisual
		virtual RenderFlags GetRenderFlags() const override;
		virtual bool IsVisible(const BoundingGeometry& viewFrustum) const override;
		virtual void Render(IRenderContext * pContext, DirectX::IEffect* pEffect = nullptr) override;
		virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;

	protected:
		// Automatic displacement by analyze joints contact ground
		void							DisplaceByVelocityFrame();
		void							GroundCharacter(float height);
		void							ComputeVelocityFrame(time_seconds time_delta);

	private:
		ISkinningModel*							m_pSkinModel;

		std::vector<Matrix4x4,
			AlignedAllocator<Matrix4x4, alignof(XMVECTOR) >>
												m_BoneTransforms;


		BehavierSpace*					        m_pBehavier;
		IArmature*								m_pArmature;
		BehavierSpace::animation_type*			m_pCurrentAction;
		BehavierSpace::animation_type*			m_pLastAction;
		time_seconds							m_CurrentActionTime;
		bool									m_LoopCurrentAction;
		LowPassFilter<float>					m_SpeedFilter;

		int										m_FrameMapState;
		frame_type						        m_CurrentFrame;
		frame_type								m_LastFrame;
		Matrix4x4								m_LastWorld;
		velocity_frame_type						m_VelocityFrame;

		std::mutex								m_ActionMutex;
		std::atomic_bool						m_UpdateLock;
		bool									m_FrameDirty;
		bool									m_IsAutoDisplacement;
	};

	class CharacterGlowParts : public GlowingBorder
	{
	public:
		CharacterGlowParts();
		typedef std::vector<DirectX::Color, DirectX::XMAllocator> BoneColorVector;
		virtual void Render(IRenderContext * pContext, DirectX::IEffect* pEffect = nullptr) override;

		virtual RenderFlags GetRenderFlags() const override;
		const	DirectX::Color& GetBoneColor(int id) const
		{ return m_BoneColors[id]; }
		void	SetBoneColor(int id, const DirectX::Color& color)
		{ m_BoneColors[id] = color; }
		// reset all bone color to Transparent
		void	ResetBoneColor(const DirectX::Color& color);
	private:
		void Initialize();

		CharacterObject*	m_pCharacter;
		BoneColorVector		m_BoneColors;
	};

	void DrawArmature(const IArmature & armature, ArmatureFrameConstView frame, const Color & color, const Matrix4x4& world = Matrix4x4::Identity, float thinkness = 0.015f);
	void DrawArmature(const IArmature & armature, ArmatureFrameConstView frame, const Color* colors, const Matrix4x4& world = Matrix4x4::Identity, float thinkness = 0.015f);
}
