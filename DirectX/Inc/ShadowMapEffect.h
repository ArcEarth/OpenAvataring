#pragma once
#include "ShaderEffect.h"

namespace DirectX
{
	class ShadowMapEffect :
		public IEffect, public IEffectMatrices, public IEffectFog, public IEffectPhongMaterial, public IEffectSkinning , public IEffectLightsShadow
	{
	public:
		ShadowMapEffect(ID3D11Device* device);
		~ShadowMapEffect();

		enum ShadowMapEffectMode
		{
			LightSpaceShadowRender = 0,
			ScreenSpaceShadowGeneration = 1,
			ScreenSpaceShadowRender = 2,
		};


		ShadowMapEffectMode GetEffectMode() const;
		void SetEffectMode(ShadowMapEffectMode mode);
		// Each channel will be ues as a shadow mask for one light, to achieve best result, blured and unblured shadow map need to be submmit 
		void __cdecl SetScreenSpaceLightsShadowMap(ID3D11ShaderResourceView* pSharpShadow, ID3D11ShaderResourceView* pSoftShadow) override;

		// Inherited via IEffect
		virtual void __cdecl Apply(ID3D11DeviceContext * deviceContext) override;
		virtual void __cdecl GetVertexShaderBytecode(void const ** pShaderByteCode, size_t * pByteCodeLength) override;

		// Inherited via IEffectMatrices
		virtual void XM_CALLCONV SetWorld(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetView(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetProjection(FXMMATRIX value) override;

		// Inherited via IEffectFog
		virtual void __cdecl SetFogEnabled(bool value) override;
		virtual void __cdecl SetFogStart(float value) override;
		virtual void __cdecl SetFogEnd(float value) override;
		virtual void XM_CALLCONV SetFogColor(FXMVECTOR value) override;

		// Inherited via IEffectPhongMaterial
		virtual void XM_CALLCONV SetDiffuseColor(FXMVECTOR value) override;
		virtual void XM_CALLCONV SetEmissiveColor(FXMVECTOR value) override;
		virtual void XM_CALLCONV SetSpecularColor(FXMVECTOR value) override;
		virtual void __cdecl SetSpecularPower(float value) override;
		virtual void __cdecl DisableSpecular() override;
		virtual void __cdecl SetAlpha(float value) override;
		virtual void __cdecl SetAlphaDiscard(bool enable) override;
		virtual void __cdecl SetDiffuseMap(ID3D11ShaderResourceView * pTexture) override;
		virtual void __cdecl SetNormalMap(ID3D11ShaderResourceView * pTexture) override;
		virtual void __cdecl SetSpecularMap(ID3D11ShaderResourceView * pTexture) override;

		// Inherited via IEffectSkinning
		static const size_t MaxBones = 72;
		virtual void __cdecl SetWeightsPerVertex(int value) override;
		virtual void __cdecl SetBoneTransforms(XMMATRIX const * value, size_t count) override;
		virtual void __cdecl ResetBoneTransforms() override;

		// Inherited via IEffectLightsShadow
		virtual void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) override;
		virtual void __cdecl SetLightEnabled(int whichLight, bool value) override;
		virtual void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) override;
		virtual void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) override;
		virtual void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) override; // Specular Color of light is not supported
		virtual void __cdecl EnableDefaultLighting() override;

		virtual void __cdecl SetLightShadowMapBias(int whichLight, float bias) override;
		virtual void __cdecl SetLightShadowMap(int whichLight, ID3D11ShaderResourceView * pTexture) override;
		virtual void XM_CALLCONV SetLightView(int whichLight, FXMMATRIX value) override;
		virtual void XM_CALLCONV SetLightProjection(int whichLight, FXMMATRIX value) override;

	private:
		class Impl;
		std::unique_ptr<Impl> pImpl;
	};

	class BinaryShadowMapEffect
		: public IEffect, public IEffectMatrices, public IEffectSkinning, public IEffectLightsShadow
	{
		// Inherited via IEffect
		virtual void __cdecl Apply(ID3D11DeviceContext * deviceContext) override;
		virtual void __cdecl GetVertexShaderBytecode(void const ** pShaderByteCode, size_t * pByteCodeLength) override;

		// Inherited via IEffectMatrices
		virtual void XM_CALLCONV SetWorld(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetView(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetProjection(FXMMATRIX value) override;

		// Inherited via IEffectSkinning
		static const size_t MaxBones = 72;
		virtual void __cdecl SetWeightsPerVertex(int value) override;
		virtual void __cdecl SetBoneTransforms(XMMATRIX const * value, size_t count) override;
		virtual void __cdecl ResetBoneTransforms() override;

		// Methods need for IEffectLightsShadow
		virtual void __cdecl SetLightEnabled(int whichLight, bool value) override = 0;
		virtual void __cdecl SetLightShadowMapBias(int whichLight, float bias) = 0;
		virtual void __cdecl SetLightShadowMap(int whichLight, ID3D11ShaderResourceView* pTexture) = 0;
		virtual void XM_CALLCONV SetLightView(int whichLight, FXMMATRIX value) = 0;
		virtual void XM_CALLCONV SetLightProjection(int whichLight, FXMMATRIX value) = 0;
	};
}