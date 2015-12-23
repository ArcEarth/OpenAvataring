#pragma once
#include "Renderable.h"
#include <VertexTypes.h>
#include <wrl\client.h>
#include <string>
#include <Effects.h>
#include "DirectXHelper.h"
#include "Textures.h"
#include "Models.h"


namespace DirectX
{
	//class CubeTexture;

	namespace Scene
	{
		class SkyDome : public IRenderable , public IViewable , protected TypedMeshBuffer<VertexPositionTexture, uint16_t>
		{
			typedef TypedMeshBuffer<VertexPositionTexture, uint16_t> BaseType;
			typedef VertexPositionTexture VertexType;
			typedef uint16_t IndexType;
			const static unsigned int VerticesCount = 24;
			const static unsigned int IndicesCount = 36;
			const static VertexType CubeVertices[VerticesCount];
			const static IndexType CubeIndices[IndicesCount];
		public:
			SkyDome(ID3D11Device* pDevice, const std::wstring(&TextureFiles)[6]);

			virtual void Render(ID3D11DeviceContext* pDeviceContext) override;
			// Inherited via IViewable
			virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;
			//virtual void XM_CALLCONV UpdateProjectionMatrix(DirectX::FXMMATRIX projection) override;

			~SkyDome(void);

		protected:
			//std::shared_ptr<MeshBuffer>			m_pMesh;
			std::shared_ptr<BasicEffect>	m_pEffect;
			std::shared_ptr<CubeTexture>	m_pCubeTexture;
			Microsoft::WRL::ComPtr<ID3D11DepthStencilState> m_pDepthStenticlState;

		};

		class ITerrain abstract
		{
			virtual Vector2 TerrainBoundary() const = 0;
			virtual float HeightAt(const Vector2& Position) const = 0;
			//virtual float HeightAt(const Vector3& Position) const = 0;
		};

		class TerrainModel : public ITerrain
		{
		};

		//class FloorPlane
		//	: public TypedMeshBuffer<VertexPositionNormalTexture, uint16_t, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST>
		//{
		//public:
		//	const static unsigned int VerticesCount = 4;
		//	const static unsigned int IndicesCount = 6;
		//	const static VertexType Vertices[VerticesCount];
		//	const static IndexType Indices[IndicesCount];

		//	FloorPlane(ID3D11Device* pDevice, const ICamera* pCamera, const std::wstring &TextureFile);
		//	~FloorPlane(void);

		//	void SetFloorTexture(ID3D11ShaderResourceView* pTextureSRV);
		//	void SetFloorPlaneEquation(DirectX::FXMVECTOR PlaneEquation);

		//	virtual void Render(ID3D11DeviceContext* pDeviceContext);

		//protected:
		//	const ICamera*										m_pCamera;
		//	std::unique_ptr<BasicEffect>						m_pEffect;
		//	Microsoft::WRL::ComPtr<ID3D11Resource>				m_pTexture;
		//	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>	m_pTextureView;
		//	Microsoft::WRL::ComPtr<ID3D11SamplerState>			m_pSamplerState;
		//	Microsoft::WRL::ComPtr<ID3D11InputLayout>			m_pInputLayout;
		//};
	}
}

