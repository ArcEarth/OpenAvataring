#pragma once
#include "VisualObject.h"
#include "Models.h"

namespace DirectX
{
	class SkydomeEffect;
}

namespace Causality
{
	class SkyDome : virtual public SceneObject, virtual public IVisual
	{
	public:
		SkyDome();
		~SkyDome();

		virtual void Parse(const ParamArchive* store) override;

		void CreateDeviceResource(ID3D11Device* device, DirectX::SkydomeEffect* pEffect);
		void SetTexture(DirectX::Texture& texture);

		// Inherited via IVisual
		virtual bool IsVisible(const DirectX::BoundingGeometry & viewFrustum) const override;

		virtual void Render(IRenderContext * context, DirectX::IEffect* pEffect = nullptr) override;

		virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;

		// Inherited via IVisual
		virtual RenderFlags GetRenderFlags() const override;
	private:
		std::shared_ptr<MeshBuffer>						m_pSphere;
		// This texture must be a cube texture
		DirectX::Texture								m_Texture;
		DirectX::SkydomeEffect*							m_pEffect;
	};
}