#pragma once
#include "VisualObject.h"
#include "RenderSystemDecl.h"
#include <HUD.h>

namespace Causality
{
	enum SpritePlacementEnum
	{
		SpritePlacement_Top,
		SpritePlacement_Bottom,
		SpritePlacement_Right,
		SpritePlacement_Left,
	};

	// A SpriteObject is a textured quad card always facing camera direction
	class SpriteObject : public SceneObject, public IVisual
	{
	public:
		SpriteObject();
		virtual bool IsVisible(const BoundingGeometry& viewFrustum) const override;
		virtual RenderFlags GetRenderFlags() const override;
		virtual void Render(IRenderContext * pContext, IEffect* pEffect = nullptr) override;
		virtual void XM_CALLCONV UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection) override;

		I2DContext* Get2DContext(IRenderDevice * pDevice = nullptr);
		I2DContext* Get2DContext(IRenderContext * pContext);
		I2DFactory* Get2DFactory();
		ITextFactory* GetTextFactory();

		//XMVECTOR GetSize() const;
		//void	 SetSize(FXMVECTOR size);

		bool	 GetBoundingBox(BoundingBox& box) const override;
		bool	 GetBoundingGeometry(BoundingGeometry& geo) const override;

		Texture2D*	GetTexture() const;
		void		SetTexture(Texture2D* texture);

	private:
		Texture2D* m_pTexture;
		SpritePlacementEnum 
				   m_placement;

		Matrix4x4  m_view;
		Matrix4x4  m_proj;

		Vector4	   m_positionProj;
		Vector2	   m_sizeProj;
	};

	// An Shared D2D Render Surface, which are rendered as 3D sprite texture in the space
	class SpriteCanvas : public SpriteObject, public DirectX::Scene::HUDCanvas
	{
	protected:
		using CanvasFacade = DirectX::Scene::HUDCanvas;
		using SpriteFacade = SpriteObject;
	private:
		using SpriteFacade::AddChild;
	public:
		using CanvasFacade::AddChild;

		SpriteCanvas();
		~SpriteCanvas();

		SpriteCanvas(IRenderDevice *pDevice, size_t width, size_t height);

		void CreateDeviceResources(IRenderDevice *pDevice, size_t width, size_t height);

		virtual void Render(IRenderContext * pContext, IEffect* pEffect = nullptr) override
		{
			if (CanvasFacade::IsDirety())
			{
				auto context = Get2DContext(pContext);
				CanvasFacade::Render(context);
			}

			SpriteFacade::Render(pContext, pEffect);
		}

	protected:
		// Decal texture for rendering highlights in target model
		sptr<PhongMaterial>			m_decalMat;
		uptr<RenderableTexture2D>	m_canvas;
	};

	// Render of floating hud is delayed to 2D render process
	// it's nothing more than a placement and visibility helper
	//class FloatingHud : public VisualObject
	//{
	//public:
	//	FloatingHud(HUDElement* hudElem);

	//	virtual void Update(time_seconds const& time_delta);

	//	virtual bool IsVisible(const BoundingGeometry& viewFrustum) const override;
	//	virtual RenderFlags GetRenderFlags() const override;
	//	virtual void Render(IRenderContext * pContext, IEffect* pEffect = nullptr) override;
	//	virtual void XM_CALLCONV UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection) override;

	//protected:
	//	Texture2D* m_pTexture;
	//	Vector2	   m_position;
	//	Vector2	   m_size;
	//	Vector2    m_viewportSize;
	//};
}