#include <ShaderEffect.h>

namespace DirectX
{
	class ShaderEffect
		: public IEffect , public IShaderEffect
	{
	public:
		ShaderEffect() {}

		ShaderEffect(_In_ ID3D11Device *pDevice, _In_ const char *VertexShaderFile, _In_opt_ const char *GeometryShaderFile , const char *PixelShaderFile)
		{
			std::string fileData = DirectX::ReadFileToString(VertexShaderFile);
			m_VertexShaderByteCode = fileData;
			m_VertexShaderByteCode.shrink_to_fit();
			//Create the VertexShader
			DirectX::ThrowIfFailed(
				pDevice->CreateVertexShader(
				fileData.c_str(),
				fileData.size(),
				nullptr,
				m_pVertexShader.GetAddressOf()
				));
			//Create the InputLayout
			m_pInputLayout = CreateInputLayout<VertexType>(pDevice, fileData.c_str(), fileData.size());

			//Create the PixelShader
			fileData = DirectX::ReadFileToString(PixelShaderFile);
			DirectX::ThrowIfFailed(
				pDevice->CreatePixelShader(
				fileData.c_str(),
				fileData.size(),
				nullptr,
				m_pPixelShader.GetAddressOf()
				));

			//Create the GeometryShader
			if (GeometryShaderFile != nullptr) {
				fileData = DirectX::ReadFileToString(GeometryShaderFile);
				DirectX::ThrowIfFailed(
					pDevice->CreateGeometryShader(
					fileData.c_str(),
					fileData.size(),
					nullptr,
					m_pGeometryShader.GetAddressOf()
					));
			}
		}

		template <size_t VertexShaderBytecodeLength, size_t PixelShaderBytecodeLength>
		ShaderEffect(_In_ ID3D11Device *pDevice, _In_ const BYTE (&VertexShaderBytecode)[VertexShaderBytecodeLength], _In_ const BYTE(&PixelShaderBytecode)[PixelShaderBytecodeLength])
		{
		}

		~ShaderEffect()
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

		virtual void __cdecl  Apply(_In_ ID3D11DeviceContext* pDeviceContext) override
		{
			pDeviceContext->VSSetShader(m_pVertexShader.Get(), nullptr, 0);
			pDeviceContext->GSSetShader(m_pGeometryShader.Get(), nullptr, 0);
			pDeviceContext->PSSetShader(m_pPixelShader.Get(), nullptr, 0);
		}

		virtual void __cdecl  GetVertexShaderBytecode(_Out_ void const** pShaderByteCode, _Out_ size_t* pByteCodeLength) override
		{
			*pShaderByteCode = m_VertexShaderByteCode.c_str();
			*pByteCodeLength = m_VertexShaderByteCode.size();
		}

	protected:
		std::string										m_VertexShaderByteCode;

		Microsoft::WRL::ComPtr<ID3D11VertexShader>		m_pVertexShader;
		Microsoft::WRL::ComPtr<ID3D11GeometryShader>	m_pGeometryShader;
		Microsoft::WRL::ComPtr<ID3D11PixelShader>		m_pPixelShader;
	};

}