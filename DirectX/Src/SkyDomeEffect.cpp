#include "pch_directX.h"
#include "SkyDomeEffect.h"

using namespace DirectX;
using namespace Internal;

#include "Shaders\Windows\SkyDomeEffect_VS.inc"
#include "Shaders\Windows\SkyDomeEffect_PS.inc"

class SkydomeEffect::Impl : public ShaderEffectBase<Internal::SkydomEffectConstantBuffer>, public AlignedNew<XMVECTOR>
{
public:
	Impl(ID3D11Device* device)
	{
		CD3D11_DEFAULT d;
		CD3D11_SAMPLER_DESC desc(d);
		device->CreateSamplerState(&desc, &m_sampler);
		CreateDeviceResource(device, SkyDomeEffect_VS, SkyDomeEffect_PS);
		m_dirty = true;
		m_envMap = nullptr;
	}

	inline void Apply(ID3D11DeviceContext * deviceContext)
	{
		if (m_dirty)
		{
			m_constants.WorldViewProj = m_world * m_view * m_proj;
			m_constants.WorldViewProj = XMMatrixTranspose(m_constants.WorldViewProj);
			m_cbuffer.SetData(deviceContext, m_constants);
			m_dirty = false;
		}
		ApplyShaders(deviceContext);
		deviceContext->PSSetShaderResources(0, 1, &m_envMap);
		deviceContext->PSSetSamplers(0, 1, &m_sampler);
	}

	Matrix4x4 m_world;
	Matrix4x4 m_view;
	Matrix4x4 m_proj;
	bool	  m_dirty;
	ID3D11ShaderResourceView *m_envMap;
	ID3D11SamplerState		 *m_sampler;
};

SkydomeEffect::SkydomeEffect(ID3D11Device * device)
	: m_pImpl(new Impl(device))
{
}

void SkydomeEffect::Apply(ID3D11DeviceContext * deviceContext)
{
	m_pImpl->Apply(deviceContext);
}

void SkydomeEffect::GetVertexShaderBytecode(void const ** pShaderByteCode, size_t * pByteCodeLength)
{
	m_pImpl->GetVertexShaderBytecode(pShaderByteCode, pByteCodeLength);
}

void XM_CALLCONV SkydomeEffect::SetWorld(FXMMATRIX value)
{
	m_pImpl->m_world = value;
	m_pImpl->m_dirty = true;
}

void XM_CALLCONV SkydomeEffect::SetView(FXMMATRIX value)
{
	m_pImpl->m_view = value;
	m_pImpl->m_dirty = true;
}

void XM_CALLCONV SkydomeEffect::SetProjection(FXMMATRIX value)
{
	m_pImpl->m_proj = value;
	m_pImpl->m_dirty = true;
}

void SkydomeEffect::SetEnviromentMap(ID3D11ShaderResourceView * pEnvMap)
{
	m_pImpl->m_envMap = pEnvMap;
}
