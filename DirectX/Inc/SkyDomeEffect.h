#pragma once
#include "ShaderEffectBase.h"
#include "DirectXMathExtend.h"

namespace DirectX
{
	namespace Internal
	{
		using namespace HLSLVectors;
		struct SkydomEffectConstantBuffer
		{
			matrix WorldViewProj;
			float4 DiffuseColor;
			float4 EmissiveColor;
		};
	}

	class SkydomeEffect : public IEffect, public IEffectMatrices
	{
	public:
		SkydomeEffect(ID3D11Device *device);
		// Inherited via IEffect
		virtual void __cdecl Apply(ID3D11DeviceContext * deviceContext) override;
		virtual void __cdecl GetVertexShaderBytecode(void const ** pShaderByteCode, size_t * pByteCodeLength) override;

		// Inherited via IEffectMatrices
		virtual void XM_CALLCONV SetWorld(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetView(FXMMATRIX value) override;
		virtual void XM_CALLCONV SetProjection(FXMMATRIX value) override;

		void __cdecl SetEnviromentMap(ID3D11ShaderResourceView* pEnvMap);
	private:
		class Impl;
		std::unique_ptr<Impl> m_pImpl;
	};
}