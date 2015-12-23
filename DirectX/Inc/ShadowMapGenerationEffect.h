#pragma once
#include "ShaderEffect.h"

namespace DirectX
{
	class ShadowMapGenerationEffect
		: public IEffect, public IEffectMatrices, public IEffectSkinning
	{
	public:
		ShadowMapGenerationEffect(ID3D11Device* device);
		~ShadowMapGenerationEffect();

		// Shadow Map setting
		enum ShadowFillMode
		{
			DepthFill = 0,
			SolidColorFill = 1,
			BoneColorFill = 2,
		};

		void __cdecl SetShadowMap(ID3D11DepthStencilView* pShaodwMap, ID3D11RenderTargetView* pRTV = NULL);
		ShadowFillMode GetShadowFillMode() const;
		void XM_CALLCONV SetShadowFillMode(_In_ ShadowFillMode mode);
		void XM_CALLCONV SetShadowColor(FXMVECTOR color = Colors::Black);
		virtual void __cdecl SetBoneColors(_In_reads_(count) XMVECTOR const* value, size_t count);

		// Use texture for alpha clipping
		void __cdecl SetAlphaDiscardThreshold(float clipThreshold);
		void __cdecl SetAlphaDiscardTexture(ID3D11ShaderResourceView* pTexture);
		void __cdecl DisableAlphaDiscard();

		// IEffectMatrices
		virtual void XM_CALLCONV SetWorld(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetView(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetProjection(FXMMATRIX value) override;

		// IEffectSkinning
		static const size_t MaxBones = 72;
		virtual void __cdecl SetWeightsPerVertex(int value) override;
		virtual void __cdecl SetBoneTransforms(_In_reads_(count) XMMATRIX const* value, size_t count) override;
		virtual void __cdecl ResetBoneTransforms() override;

		// Inherited via IEffect
		virtual void __cdecl Apply(ID3D11DeviceContext * deviceContext) override;
		virtual void __cdecl GetVertexShaderBytecode(void const ** pShaderByteCode, size_t * pByteCodeLength) override;
	private:
		class Impl;
		std::unique_ptr<Impl> pImpl;
	};

}