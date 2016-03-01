#pragma once
#include "Causality\CharacterObject.h"
#include "Causality\KinectSensor.h"
#include "Causality\Animations.h"
#include "Causality\Interactive.h"

#include "CharacterController.h"
#include "PlayerSelector.h"
#include <atomic>

namespace Causality
{
	//using boost::circular_buffer;

	// Player : public Character
	class PlayerProxy : public SceneObject, public IVisual, public IAppComponent, public IKeybordInteractive
	{
	public:
#pragma region Constants
		static const size_t	FrameRate = ANIM_STANDARD::SAMPLE_RATE;
		static const size_t	ScaledMotionTime = ANIM_STANDARD::MAX_CLIP_DURATION; // second
#pragma endregion

		// Character Map State
		bool							IsMapped() const;

		const CharacterController&		CurrentController() const;
		CharacterController&			CurrentController();
		const CharacterController&		GetController(int state) const;
		CharacterController&			GetController(int state) ;

		virtual void					OnKeyUp(const KeyboardEventArgs&e) override;
		virtual void					OnKeyDown(const KeyboardEventArgs&e) override;

		// SceneObject interface
		PlayerProxy();
		virtual		 ~PlayerProxy() override;
		virtual void AddChild(SceneObject* pChild) override;
		virtual void Parse(const ParamArchive* store) override;

		// Render / UI Thread 
		void Update(time_seconds const& time_delta) override;
		void UpdateSelfMotionBinder(const time_seconds & time_delta);

		void UpdateThreadRuntime();
		void StartUpdateThread();
		void StopUpdateThread();

		// Inherited via IVisual
		virtual bool IsVisible(const DirectX::BoundingGeometry & viewFrustum) const override;
		virtual void Render(IRenderContext * context, DirectX::IEffect* pEffect = nullptr) override;
		virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;

		// PlayerSelector Interface
		const IPlayerSelector&		GetPlayerSelector() const { return *m_pSelector; }
		IPlayerSelector&			GetPlayerSelector() { return *m_pSelector; }
		void						SetPlayerSelector(const sptr<IPlayerSelector>& playerSelector);

		const IArmature&			Armature() const { return *m_pPlayerArmature; };
		const ShrinkedArmature&		Parts() const { return *m_pParts; }

	protected:

		void	UpdatePrimaryCameraForTrack();
		void	ResetPrimaryCameraPoseToDefault();

		// Helper methods
		//bool	UpdateByFrame(ArmatureFrameConstView frame);

		void	SetActiveController(int idx);
		int		MapCharacterByLatestMotion();

		// plyaer streaming thread
		friend	IPlayerSelector;
		void	StreamPlayerFrame(const IArmatureStreamAnimation& body, const IArmatureStreamAnimation::FrameType& frame);
		void	ResetPlayer(IArmatureStreamAnimation* pOld, IArmatureStreamAnimation* pNew);

		void	ResetPlayerArmature(const IArmature* playerArmature);
		void	InitializeShrinkedPlayerArmature();
		//void	ClearPlayerFeatureBuffer();

	protected:
		bool								m_EnableOverShoulderCam;
		bool								m_IsInitialized;

		DirectX::Texture2D*					m_trailVisual;

		std::thread							m_updateThread;
		std::atomic_bool					m_stopUpdate;
		std::mutex							m_controlMutex;
		std::atomic_bool					m_mapTaskOnGoing;
		std::atomic_bool					m_newFrameAvaiable;
		std::atomic_int						m_updateCounter;
		concurrency::task<void>				m_mapTask;

		const IArmature*					m_pPlayerArmature;

		uptr<ShrinkedArmature>				m_pParts;
		vector<Color, DirectX::XMAllocator>	m_boneColors;
		int									m_Id;

		sptr<IPlayerSelector>				m_pSelector;

		ArmatureFrame						m_pushFrame;

		double								m_updateTime;
		std::chrono::time_point<std::chrono::system_clock> 
											m_lastUpdateTime;
		ArmatureFrame						m_currentFrame;
		ArmatureFrame						m_lastFrame;

		double								m_LowLikilyTime;
		CyclicStreamClipinfo				m_CyclicInfo;

		int									m_CurrentIdx;
		std::list<CharacterController>		m_Controllers;

		bool								m_DefaultCameraFlag;
		RigidTransform						m_DefaultCameraPose;

		time_seconds						current_time;

	protected:
		// Enter the selecting phase
		//void BeginSelectingPhase();
		// End the selecting phase and enter the manipulating phase
		//void BeginManipulatingPhase();

		//std::pair<float, float> ExtractUserMotionPeriod();

		// Inherited via IVisual
		virtual RenderFlags GetRenderFlags() const override;
		//void PrintFrameBuffer(int No);
	};

	class KinectVisualizer : public SceneObject, public IVisual
	{
	public:
		KinectVisualizer();
		// Inherited via IVisual
		virtual RenderFlags GetRenderFlags() const override;
		virtual bool IsVisible(const DirectX::BoundingGeometry & viewFrustum) const override;
		virtual void Render(IRenderContext * context, DirectX::IEffect* pEffect = nullptr) override;
		virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;

	protected:
		std::shared_ptr<Devices::KinectSensor>	pKinect;

	};
}

void DrawParticle(DirectX::XMVECTOR &particle, const DirectX::XMMATRIX &world, const DirectX::CXMMATRIX &proj, const D3D11_VIEWPORT & vp, DirectX::SpriteBatch & sprites, ID3D11ShaderResourceView * pTrajectoryVisual, const DirectX::XMVECTOR &color);
