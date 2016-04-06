#pragma once
#include "RenderSystemDecl.h"
#include "Math3D.h"
#include "SceneObject.h"
#include "RenderFlags.h"
#include "SmartPointers.h"
#include <Textures.h>

namespace DirectX
{
	class IPostEffect;
	class ShadowMapEffect;
}

namespace Causality
{

	namespace Devices
	{
		class OculusRift;
	}

	struct CameraBuffer
	{
		Matrix4x4	ViewProjectionMatrix;
		Vector4		CameraPosition;
		Vector4		CameraFocus;
		Matrix4x4	ViewMatrix;
		Matrix4x4	ProjectionMatrix;
	};

	// The basic functions for camera, to provide View/Projection Matrix, Setup Position and focus
	class IViewControl abstract
	{
	public:
		virtual XMMATRIX						GetViewMatrix() const = 0;
		virtual XMMATRIX						GetProjectionMatrix() const = 0;
		virtual void							FocusAt(FXMVECTOR focusPoint, FXMVECTOR upDir) = 0;
		virtual const BoundingGeometry&			GetViewFrustum() const = 0;

		//XMVECTOR								GetFowardDir() const {}
		//XMVECTOR								GetUpDir() const {}
		//XMVECTOR								GetRightDir() const {}
		//XMVECTOR								GetFocusDir() const { return GetFowardDir(); }
		//virtual XMVECTOR						GetFocusPosition() const { return GetFocusDir(); };
		//XMVECTOR								GetViewCenter() const {}

		inline XMMATRIX				GetViewProjectionMatrix() const
		{
			XMMATRIX mat = GetViewMatrix();
			mat *= GetProjectionMatrix();
			return mat;
		}
	};

	// Control the logic of render target setup and post-processing needed for current camera
	class IRenderControl abstract
	{
	public:
		// Called in the beginning of per-frame
		virtual void Begin(IRenderContext* context) = 0;
		// Called in in the end of per frame, should call Prenset inside
		virtual void End() = 0;

		virtual IEffect* GetRenderEffect() = 0;
		virtual IPostEffect* GetPostEffect() = 0;
		virtual RenderTarget& GetRenderTarget() = 0;
		virtual Texture2D& GetOutput() = 0;

		virtual bool AcceptRenderFlags(RenderFlags flags) = 0;
	};

	class ICamera
	{
	public:
		virtual size_t ViewCount() const = 0;
		virtual IViewControl* GetView(int view = 0) = 0;
		virtual size_t ViewRendererCount(int view = 0) const = 0;
		virtual IRenderControl* GetViewRenderer(int view = 0, int renderer = 0) = 0;

		virtual void	BeginFrame() = 0;
		virtual void	EndFrame() = 0;

		// Focus all View at
		inline void FocusAt(FXMVECTOR focusPoint, FXMVECTOR upDir)
		{
			for (int i = 0; i < ViewCount(); i++)
			{
				GetView(i)->FocusAt(focusPoint, upDir);
			}
		}

		inline  const IViewControl* GetView(int view = 0) const { return const_cast<ICamera*>(this)->GetView(view); }
	};

	class CameraViewControl : virtual public IViewControl
	{
	public:
		CameraViewControl();
		virtual XMMATRIX		GetViewMatrix() const override;
		virtual XMMATRIX		GetProjectionMatrix() const override;
		virtual void			FocusAt(FXMVECTOR focusPoint, FXMVECTOR upDir) override;
		virtual const BoundingGeometry&	GetViewFrustum() const override;

		// Tilt (Rotate) the camera along 'forward' direction, 'roll' direction
		void	Tilt(float rad);

		IRigid* GetAttachedRigid() const { return m_Parent; }
		void	SetAttachedRigid(IRigid* pRigid);

		void	SetBiasdPerspective();
		void	SetPerspective(float fovRadius, float aspectRatioHbyW, float Near = 0.01f, float Far = 100.0f);
		void	SetOrthographic(float viewWidth, float viewHeight, float Near = 0.01f, float Far = 100.0f);
		void	SetHandness(bool rightHand);

		bool	IsPerspective() const;
		float	GetFov() const;
		float	GetAspectRatio() const;
		float	GetNear() const;
		float	GetFar() const;
		void	SetFov(float fov);
		void	SetAspectRatio(float aspectHbyW);
		void	SetNear(float _near);
		void	SetFar(float _far);

		XMVECTOR GetDisplacement() const;
		void	SetDisplacement(FXMVECTOR displacement);

		static const XMVECTORF32 Foward, Up;
	private:
		IRigid*										m_Parent;
		Vector3										m_Displacement; // local transform form view to camera
		Vector3										m_Focus; // Relative focus point
		Vector3										m_UpDir;
		bool										m_IsRightHand;
		bool										m_IsPerspective;
		float										m_Near, m_Far;
		float										m_Fov, m_AspectRatio;

		mutable BoundingGeometry					m_ExtrinsicViewFrutum; // the frutum also transformed by View Matrix
		mutable BoundingGeometry					m_ViewFrutum; // the frutum defined by Projection Matrix
		mutable Matrix4x4							m_ViewCache; // extrinsic matrix
		mutable Matrix4x4							m_ProjectionCache; // instrinsic matrix
		mutable bool								m_ViewDirty;
		mutable bool								m_ProjectDirty;

		XMMATRIX UpdateProjectionCache() const;
	};

	// A Effect camera always follow it's owner's view and projection, but have different render control
	class EffectRenderControl : virtual public IRenderControl
	{
	public:
		EffectRenderControl();

		~EffectRenderControl();

		// Inherited via IRenderControl
		virtual void Begin(IRenderContext* context) override;
		virtual void End() override;
		virtual bool AcceptRenderFlags(RenderFlags flags) override;
		// Only need for Stereo or more camera
		void SetView(IViewControl *pViewControl);

		virtual IEffect* GetRenderEffect();
		virtual IPostEffect* GetPostEffect();

		void SetRequestRenderFlags(RenderFlags flags);

		void SetRenderEffect(const std::shared_ptr<IEffect>& pEffect);
		void SetPostEffect(const std::shared_ptr<IPostEffect>& pPostEffect);

		void	SetRenderTargetClearence(bool ifClear);
		Color	GetBackground() const;
		void	SetBackground(const Color& color);


		virtual RenderTarget&	GetRenderTarget() override;
		virtual Texture2D&		GetOutput() override;

		void SetRenderTarget(RenderTarget & renderTarget);
		void SetPostEffectOutput(RenderableTexture2D& output);

	protected:
		bool										m_IfClearRenderTarget;
		bool										m_HaveItemRendered;
		RenderFlags									m_RequstFlags;
		Color										m_Background;
		cptr<IRenderContext>						m_pRenderContext;
		sptr<IEffect>								m_pEffect;
		sptr<IPostEffect>							m_pPostEffect;
		RenderTarget								m_RenderTarget;
		RenderableTexture2D							m_PostEffectOutput;
	};

	class SingleViewCamera : public SceneObject, public virtual ICamera, public CameraViewControl
	{
	public:
		SingleViewCamera();

		virtual void Parse(const ParamArchive* store) override;

		virtual void CreateDeviceResources(IRenderDevice* pDevice, RenderTarget& canvas) = 0;

		virtual size_t ViewCount() const override;
		virtual IViewControl* GetView(int view = 0) override;

		// A direct call to monolith camera's focus will change camera's orientation instead of CameraViewControl's local coords
		virtual void FocusAt(FXMVECTOR focusPoint, FXMVECTOR upDir) override;

		virtual void BeginFrame() override;
		virtual void EndFrame() override;
	};

	// Standard camera with single viewport and single rendering pass
	class Camera : public SingleViewCamera, public EffectRenderControl
	{
	public:
		Camera();

		virtual void CreateDeviceResources(IRenderDevice* pDevice, RenderTarget& canvas) override;

		virtual size_t ViewRendererCount(int view = 0) const override;
		virtual IRenderControl* GetViewRenderer(int view = 0, int renderer = 0) override;

		void SetRenderTarget(RenderTarget & renderTarget);
	};

	class MultipassCamera : public SingleViewCamera
	{
	public:
		MultipassCamera();
		virtual size_t ViewRendererCount(int view = 0) const;
		virtual IRenderControl* GetViewRenderer(int view = 0, int renderer = 0);

		void AddEffectRenderControl(const shared_ptr<EffectRenderControl>& pRenderer);
	protected:
		vector<shared_ptr<EffectRenderControl>> m_pRenderers;
	};

	class MuiltiviewCamera : public SceneObject, public ICamera
	{
	public:
		virtual size_t ViewCount() const;
		virtual IViewControl* GetView(int view = 0);
		virtual size_t ViewRendererCount(int view = 0) const;
		virtual IRenderControl* GetViewRenderer(int view = 0, int renderer = 0);
		virtual void BeginFrame() override;
		virtual void EndFrame() override;

		void AddEffectRenderControl(const shared_ptr<EffectRenderControl>& pRenderer);
		void AddViewRenderControl(int view , int rendererIdx);
	protected:
		vector<unique_ptr<CameraViewControl>>	m_Views;
		vector<RenderTarget>					m_RenderTargets;
		vector<vector<unsigned>>				m_ViewportRenderers;
		vector<shared_ptr<EffectRenderControl>>	m_pRenderers;
	};

	class SoftShadowCamera : public MultipassCamera
	{
	public:
		SoftShadowCamera();
		void CreateDeviceResources(IRenderDevice* pDevice, RenderTarget& canvas) override;
	};

	class PercentCloserShadowCamera : public MultipassCamera
	{
	public:
		PercentCloserShadowCamera();
		void CreateDeviceResources(IRenderDevice* pDevice, RenderTarget& canvas) override;
	};

	class HMDCamera : public MuiltiviewCamera
	{
	public:
		HMDCamera();
		~HMDCamera();

		void CreateDeviceResources(IRenderDevice* pDevice, RenderTarget& canvas);

		virtual void Parse(const ParamArchive* store) override;
		virtual void BeginFrame() override;
		virtual void EndFrame() override;

		float GetIPD() const { return m_ipd; }
		void SetIPD(float ipd);

		// Automatic optimze projection matrix for left/right eye
		void SetPerspective(float fovRadius, float aspectRatioHbyW, float Near = 0.01f, float Far = 100.0f);
		void SetOrthographic(float viewWidth, float viewHeight, float Near = 0.01f, float Far = 100.0f);

	private:
		sptr<Devices::OculusRift>	m_pRift;
		cptr<IRenderContext>		m_pContext;
		float						m_ipd; // default to 64mm
	};
}
