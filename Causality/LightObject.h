#pragma once
#include "CameraObject.h"
#include "VisualObject.h"
#include <Textures.h>

namespace Causality
{
	class ILight abstract
	{
	public:
		virtual Color GetColor() const = 0;
		virtual ID3D11ShaderResourceView* GetColorMap() const { return nullptr; }
		virtual ID3D11ShaderResourceView* GetShadowMap() const { return nullptr; }

		// Brightness is the among of color showed when the surface is UNIT distant with the light
		inline float	GetBrightness() const { return GetColor().w; }
		inline bool	DropsShadow() const { return GetShadowMap() != nullptr; }
		DirectX::XMVECTOR GetFocusDirection() const;

		inline IViewControl* AsViewControl()
		{
			return dynamic_cast<IViewControl*>(this);
		}
		inline const IViewControl* AsViewControl() const
		{
			return dynamic_cast<const IViewControl*>(this);
		}

	};
	// class to introduce a parallel or point (perspective) light
	class Light : public Camera, public ILight, public IVisual
	{
	public:
		Light();
		Light(IRenderDevice* deivce, const UINT& shadowResolution = 1024U);

		virtual void Parse(const ParamArchive* store) override;

		// using Camera to implement IViewControl interface
		using Camera::ViewCount;
		using Camera::GetViewMatrix;
		using Camera::GetProjectionMatrix;
		using Camera::FocusAt;
		using Camera::GetViewFrustum;

		// IRenderControl
		virtual void				Begin(IRenderContext* context) override;
		virtual void				End() override;
		virtual DirectX::IEffect*	GetRenderEffect() override;
		virtual bool				AcceptRenderFlags(RenderFlags flags) override;

		virtual Color	GetColor() const override;
		void	SetColor(const Color& color) { m_Color = color; }

		// Brightness is the among of color showed when the surface is UNIT distant with the light
		void	SetBrightness(float bright) { m_Color.w = bright; }

		void	DisableDropShadow();
		void	EnableDropShadow(IRenderDevice* deivce, const UINT& shadowResolution = 1024U);
		void	SetShadowMapBuffer(const DirectX::DepthStencilBuffer& depthBuffer);
		virtual ID3D11ShaderResourceView* GetShadowMap() const override;

		void	SetColorMap(ID3D11ShaderResourceView* lightMap) { m_pColorMap = lightMap; }
		virtual ID3D11ShaderResourceView* GetColorMap() const override;

		// Inherited via IVisual, for Light source Visualization
		virtual RenderFlags GetRenderFlags() const;
		virtual bool IsVisible(const BoundingGeometry & viewFrustum) const override;
		virtual void Render(IRenderContext * context, DirectX::IEffect* pEffect = nullptr) override;
		virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;
	private:
		DirectX::Color						m_Color;
		cptr<ID3D11ShaderResourceView>		m_pColorMap;
		sptr<DirectX::IEffect>				m_pShadowMapGenerator;

	};
}
