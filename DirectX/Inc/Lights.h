#ifndef LIGHTS_H
#define LIGHTS_H
#pragma once
#include "DirectXHelper.h"
#include "DirectXMathExtend.h"

namespace DirectX
{
	struct LightBuffer
	{
		static const unsigned int SphereLightCount = 4;

		DirectX::XMFLOAT4 AmbientColor;
		DirectX::XMFLOAT4 DiffuseColor;
		DirectX::XMFLOAT3 DiffuseDirection;
		float padding;
		DirectX::XMFLOAT4 SphereLightPosition[SphereLightCount];
		DirectX::XMFLOAT4 SphereLightColor[SphereLightCount];
		DirectX::XMFLOAT3 EyePosition;
		float padding1;
	};

	struct AmbientLightData
	{
		DirectX::Vector4 Color;
	};

	struct DiffuseLightData
	{
		DirectX::Vector4 Color;
		DirectX::Vector3 Direction;
	};

	struct SphereLightData
	{
		DirectX::Vector4 Color;
		DirectX::Vector3 Position;
		float Radius;
	};


	// Class for manage lights
	// Use is to 
	class LightsBuffer
	{
	private:
		UINT			m_Slot;
		ID3D11Buffer	*m_pGPUBuffer;
		LightBuffer		m_Buffer;
	public:
		AmbientLightData			AmbientLight;
		DiffuseLightData			DiffuseLight;
		SphereLightData				SphereLights[LightBuffer::SphereLightCount];
		DirectX::XMFLOAT3			EyePosition;
	public:
		LightsBuffer(ID3D11Device *pDevice, UINT LightsBufferSlot = 0) {
			m_pGPUBuffer = DirectX::CreateConstantBuffer<LightBuffer>(pDevice);
			m_Slot = LightsBufferSlot;
		}

		ID3D11Buffer* GetBuffer() { return m_pGPUBuffer; }
		ID3D11Buffer* UpdateBuffer(ID3D11DeviceContext *pContext) {
			m_Buffer.AmbientColor = AmbientLight.Color;
			m_Buffer.DiffuseColor = DiffuseLight.Color;
			m_Buffer.DiffuseDirection = DiffuseLight.Direction;
			m_Buffer.EyePosition = EyePosition;
			for (size_t i = 0; i < LightBuffer::SphereLightCount; i++)
			{
				//float invRadius = 1.0f / SphereLights[i].Radius;
				m_Buffer.SphereLightPosition[i] = reinterpret_cast<Vector4&>(SphereLights[i].Position, SphereLights[i].Radius);
				m_Buffer.SphereLightColor[i] = SphereLights[i].Color;
			}
			DirectX::SetBufferData(m_pGPUBuffer, pContext, m_Buffer);
			return m_pGPUBuffer;
		}

		void Render(ID3D11DeviceContext *pContext) {
			UpdateBuffer(pContext);
			pContext->PSSetConstantBuffers(m_Slot, 1, &m_pGPUBuffer);
		}
	};

	class ILight
	{
		virtual XMVECTOR GetColor() const;
		virtual XMMATRIX GetLightMatrix() const;
		virtual XMMATRIX GetProjectionMatrix() const;
	};
}

#endif // !LIGHTS_H
