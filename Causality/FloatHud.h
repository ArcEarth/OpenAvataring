#pragma once
#include "VisualObject.h"
#include "RenderSystemDecl.h"

namespace Causality
{
	enum FloatingHudPlacement
	{
		Top,
		Bottom,
		Right,
		Left,
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

		//XMVECTOR GetSize() const;
		//void	 SetSize(FXMVECTOR size);

		bool	 GetBoundingBox(BoundingBox& box) const override;
		bool	 GetBoundingGeometry(BoundingGeometry& geo) const override;

		Texture2D*	GetTexture() const;
		void		SetTexture(Texture2D* texture);
	private:
		Texture2D* m_pTexture;

		Vector4	   m_positionProj;
		Vector2	   m_sizeProj;
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