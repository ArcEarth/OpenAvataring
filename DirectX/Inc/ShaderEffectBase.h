#pragma once
#include "ShaderEffect.h"
#include "ConstantBuffer.h"
#include "DirectXHelper.h"

namespace DirectX
{
	namespace Internal
	{
		// Points to a precompiled vertex or pixel shader program.
		struct ShaderBytecode
		{
			void const* code;
			size_t length;
		};
	}

	template <class ConstantBufferType>
	class ShaderEffectBase
	{
	public:
		ShaderEffectBase() {}

		void CreateDeviceResource(_In_ ID3D11Device *pDevice, _In_ const char *VertexShaderFile, _In_ const char *PixelShaderFile, _In_opt_ const char *GeometryShaderFile = nullptr)
		{
			m_cbuffer.Create(pDevice);
			//Create the PixelShader
			m_fileData = DirectX::ReadData(PixelShaderFile);
			DirectX::ThrowIfFailed(
				pDevice->CreatePixelShader(
				fileData.c_str(),
				fileData.size(),
				nullptr,
				m_pPixelShader.GetAddressOf()
				));

			//Create the GeometryShader
			if (GeometryShaderFile != nullptr) {
				m_fileData = DirectX::ReadData(GeometryShaderFile);
				DirectX::ThrowIfFailed(
					pDevice->CreateGeometryShader(
					fileData.c_str(),
					fileData.size(),
					nullptr,
					m_pGeometryShader.GetAddressOf()
					));
			}

			m_fileData = DirectX::ReadData(VertexShaderFile);
			//Create the VertexShader
			DirectX::ThrowIfFailed(
				pDevice->CreateVertexShader(
					fileData.c_str(),
					fileData.size(),
					nullptr,
					m_pVertexShader.GetAddressOf()
					));
			m_vertexShaderByteCode.code = m_fileData.data();
			m_vertexShaderByteCode.length = m_fileData.size();
		}

		template <size_t VertexShaderBytecodeLength, size_t PixelShaderBytecodeLength>
		void CreateDeviceResource(_In_ ID3D11Device *pDevice, _In_ const BYTE (&VertexShaderBytecode)[VertexShaderBytecodeLength], _In_ const BYTE(&PixelShaderBytecode)[PixelShaderBytecodeLength])
		{
			m_cbuffer.Create(pDevice);

			m_vertexShaderByteCode.length = VertexShaderBytecodeLength;
			m_vertexShaderByteCode.code = VertexShaderBytecode;

			DirectX::ThrowIfFailed(
				pDevice->CreateVertexShader(
					VertexShaderBytecode,
					VertexShaderBytecodeLength,
					nullptr,
					m_pVertexShader.GetAddressOf()
					));

			//Create the PixelShader
			DirectX::ThrowIfFailed(
				pDevice->CreatePixelShader(
					PixelShaderBytecode,
					PixelShaderBytecodeLength,
					nullptr,
					m_pPixelShader.GetAddressOf()
					));

			////Create the GeometryShader
			//if (GeometryShaderFile != nullptr) {
			//	fileData = DirectX::ReadData(GeometryShaderFile);
			//	DirectX::ThrowIfFailed(
			//		pDevice->CreateGeometryShader(
			//			fileData.c_str(),
			//			fileData.size(),
			//			nullptr,
			//			m_pGeometryShader.GetAddressOf()
			//			));
			//}
		}

		~ShaderEffectBase()
		{
		}

		ID3D11VertexShader* GetVertexShader() const
		{
			return m_pVertexShader.Get();
		}
		ID3D11GeometryShader* GetGeometryShader() const
		{
			return m_pGeometryShader.Get();
		}
		ID3D11PixelShader* GetPixelShader() const
		{
			return m_pPixelShader.Get();
		}

		// Applay Shaders and Constant buffer to context
		void ApplyShaders(_In_ ID3D11DeviceContext* pDeviceContext)
		{
			pDeviceContext->VSSetShader(m_pVertexShader.Get(), nullptr, 0);
			auto cbuffer = m_cbuffer.GetBuffer();
			pDeviceContext->VSSetConstantBuffers(0, 1,&cbuffer);
			if (m_pGeometryShader)
			{
				pDeviceContext->GSSetConstantBuffers(0, 1, &cbuffer);
				pDeviceContext->GSSetShader(m_pGeometryShader.Get(), nullptr, 0);
			}
			pDeviceContext->PSSetShader(m_pPixelShader.Get(), nullptr, 0);
			pDeviceContext->PSSetConstantBuffers(0, 1, &cbuffer);
		}

		void GetVertexShaderBytecode(_Out_ void const** pShaderByteCode, _Out_ size_t* pByteCodeLength)
		{
			*pShaderByteCode = m_vertexShaderByteCode.code;
			*pByteCodeLength = m_vertexShaderByteCode.length;
		}

	protected:
		Internal::ShaderBytecode						m_vertexShaderByteCode;
		// owner of the vs shader byte code, if read from file
		std::vector<byte>								m_fileData;
		ConstantBufferType								m_constants;
		ConstantBuffer<ConstantBufferType>				m_cbuffer;
		Microsoft::WRL::ComPtr<ID3D11VertexShader>		m_pVertexShader;
		Microsoft::WRL::ComPtr<ID3D11GeometryShader>	m_pGeometryShader;
		Microsoft::WRL::ComPtr<ID3D11PixelShader>		m_pPixelShader;
	};

}