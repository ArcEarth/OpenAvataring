#pragma once
#include <DirectXMath.h>
#include <d3d11_1.h>
#include <memory>
#include "Textures.h"

namespace DirectX
{
	enum PostEffectOutputMode
	{
		Default = 0,
		AlphaBlendNoDepth = 1,
		AlphaAsDepth = 2,
		Customized = 3,
		BloomCombination = 4,
	};	
	
	// Interface for Post processing effct
	class IPostEffect
	{
	public :
		virtual ~IPostEffect() {}

		virtual void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target, const Texture2D& source, const D3D11_VIEWPORT & outputviewport, const D3D11_VIEWPORT & inputViewport) = 0;
		virtual void SetAddtionalInputResources(UINT numSource, ID3D11ShaderResourceView * const * pSources) = 0;
		virtual void SetOutputMode(PostEffectOutputMode outputMode) = 0;


		// Apply effect in-place
		void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target, const D3D11_VIEWPORT & viewport)
		{
			assert(target.ShaderResourceView() != nullptr);
			Apply(pContext, target, target, viewport, viewport);
		}

		// Apply effect in-place with full region
		void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target)
		{
			CD3D11_VIEWPORT viewport(.0f, .0f, (float)target.Width(), (float)target.Height());
			Apply(pContext, target, viewport);
		}
	};



	class ChainingPostEffect : public IPostEffect
	{
	public:
		virtual void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target, const Texture2D& source, const D3D11_VIEWPORT & outputviewport, const D3D11_VIEWPORT & inputViewport) override;
		virtual void SetAddtionalInputResources(UINT numSource, ID3D11ShaderResourceView * const * pSources) override;
		std::vector<std::shared_ptr<IPostEffect>> Effects;
	};
	
	class PassbyEffect : public IPostEffect
	{
	public:
		virtual void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target, const Texture2D& source, const D3D11_VIEWPORT & outputviewport, const D3D11_VIEWPORT & inputViewport) override;
	private:
		class Impl;
		std::unique_ptr<Impl> pImpl;
	};

	class GuassianBlurEffect : public IPostEffect
	{
	public:
		GuassianBlurEffect(ID3D11Device* pDevice);
		~GuassianBlurEffect();

		void SetOutputMode(PostEffectOutputMode outputMode);
		void SetOutputDepthStencil(ID3D11DepthStencilView* pDepthStencilBuffer);

		// aka. standard derviation , use radius = 1 to represent a 3x3 kernal
		void SetMultiplier(float multiplier);
		void SetBlurRadius(float radius);
		void ResizeBufferRespectTo(ID3D11Device* pDevice, const Texture2D & texture);
		void ResizeBuffer(ID3D11Device* pDevice, const XMUINT2& size, DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM);

		virtual void SetAddtionalInputResources(UINT numSource, ID3D11ShaderResourceView * const * pSources);
		virtual void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target, const Texture2D& source, const D3D11_VIEWPORT & outputviewport, const D3D11_VIEWPORT & inputViewport) override;
	private:
		class Impl;
		std::unique_ptr<Impl> pImpl;
	};


}