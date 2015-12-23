#pragma once
#if defined(_XBOX_ONE) && defined(_TITLE)
#include <d3d11_x.h>
#else
#include <d3d11_1.h>
#endif
#include <wrl\client.h>
#include <Effects.h>
#include <ppltasks.h>
#include <memory>

namespace DirectX
{
	class IEffectPhongMaterial
	{
	public:
		// Material settings.
		virtual void XM_CALLCONV SetDiffuseColor(FXMVECTOR value) = 0;
		virtual void XM_CALLCONV SetEmissiveColor(FXMVECTOR value) = 0;
		virtual void XM_CALLCONV SetSpecularColor(FXMVECTOR value) = 0;
		virtual void __cdecl SetSpecularPower(float value) = 0;
		virtual void __cdecl DisableSpecular() = 0;
		virtual void __cdecl SetAlpha(float value) = 0;
		virtual void __cdecl SetAlphaDiscard(bool enable) = 0;

		virtual void __cdecl SetDiffuseMap(ID3D11ShaderResourceView* pTexture) = 0;
		virtual void __cdecl SetNormalMap(ID3D11ShaderResourceView* pTexture) = 0;
		virtual void __cdecl SetSpecularMap(ID3D11ShaderResourceView* pTexture) = 0;
	};

	class IEffectLightsShadow abstract : public IEffectLights
	{
	public:
		// Lighting and per pixel lighting is always enabled for Shadowed Light Effects
		virtual void __cdecl SetLightingEnabled(bool value) {};
		virtual void __cdecl SetPerPixelLighting(bool value) {};
		virtual void XM_CALLCONV SetAmbientLightColor(FXMVECTOR value) {};
		virtual void XM_CALLCONV SetLightDirection(int whichLight, FXMVECTOR value) {};
		virtual void XM_CALLCONV SetLightDiffuseColor(int whichLight, FXMVECTOR value) {};
		virtual void XM_CALLCONV SetLightSpecularColor(int whichLight, FXMVECTOR value) {};

		// Methods need for IEffectLightsShadow
		virtual void __cdecl SetScreenSpaceLightsShadowMap(ID3D11ShaderResourceView* pSharpShadow, ID3D11ShaderResourceView* pSoftShadow) = 0;
		virtual void __cdecl SetLightEnabled(int whichLight, bool value) override = 0;
		virtual void __cdecl SetLightShadowMapBias(int whichLight, float bias) = 0;
		virtual void __cdecl SetLightShadowMap(int whichLight, ID3D11ShaderResourceView* pTexture) = 0;
		virtual void XM_CALLCONV SetLightView(int whichLight, FXMMATRIX value) = 0;
		virtual void XM_CALLCONV SetLightProjection(int whichLight, FXMMATRIX value) = 0;
	};

	// An base class for customized shader effect, will handel the loading of shaders
	class IShaderEffect
	{
	public:
		virtual ~IShaderEffect() { }
		virtual ID3D11InputLayout*		GetInputLayout() const = 0;
		virtual ID3D11VertexShader*		GetVertexShader() const = 0;
		virtual ID3D11GeometryShader*	GetGeometryShader() const = 0;
		virtual ID3D11PixelShader*		GetPixelShader() const = 0;
	};

}
