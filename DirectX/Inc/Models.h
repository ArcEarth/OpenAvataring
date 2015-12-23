#pragma once
#include "stride_range.h"
#include "DirectXHelper.h"
#include "DXGIFormatHelper.h"
#include <vector>
#include <string>
#include <Effects.h>
#include <type_traits>
#include <VertexTypes.h>
#include <DirectXCollision.h>
#include "Textures.h"
#include "Material.h"
#include "ConstantBuffer.h"
#include "Locatable.h"

namespace DirectX
{
	namespace Scene
	{
		namespace FacetPrimitives
		{
			template <class _TIndex>
			struct Triangle
			{
				union
				{
					_TIndex V[3];
					struct
					{
						_TIndex V0, V1, V2;
					};
				};

				inline _TIndex& operator[](size_t idx)
				{
					return V[idx];
				}

				inline _TIndex operator[](size_t idx) const
				{
					return V[idx];
				}

				operator XMUINT3() const
				{
					return XMUINT3(V0, V1, V2);
				}
			};
		}

		using Microsoft::WRL::ComPtr;
		// Example
		//{ "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "NORMAL",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "TANGENT",     0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "COLOR",       0, DXGI_FORMAT_R8G8B8A8_UNORM,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "BLENDINDICES",0, DXGI_FORMAT_R8G8B8A8_UINT,      0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		//{ "BLENDWEIGHT", 0, DXGI_FORMAT_R8G8B8A8_UNORM,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		struct VertexDescription
		{
			VertexDescription()
				:NumElements(0), pInputElementDescs(nullptr)
			{}

			VertexDescription(UINT num, const D3D11_INPUT_ELEMENT_DESC* pDescs) 
				: NumElements(num), pInputElementDescs(pDescs)
			{}

			UINT NumElements;
			const D3D11_INPUT_ELEMENT_DESC* pInputElementDescs;

			bool HasSemantic(LPCSTR semanticName) const
			{
				return std::find_if(pInputElementDescs, pInputElementDescs + NumElements, [semanticName](const D3D11_INPUT_ELEMENT_DESC& desc)
				{
					return !strcmp(desc.SemanticName, semanticName);
				}) != pInputElementDescs + NumElements;
			}

			bool HasNormal() const { return HasSemantic("NORMAL"); }
			bool HasColor() const { return HasSemantic("COLOR"); }
			bool HasUV() const { return HasSemantic("TEXCOORD"); }
			bool HasTagent() const { return HasSemantic("TANGENT"); }
			bool HasBinormal() const { return HasSemantic("BINORMAL"); }
			bool HasSkinningWeights() const { return HasSemantic("BLENDINDICES") && HasSemantic("BLENDWEIGHT"); }
		};

		// A Container of Vertex and Indices holding geometry information with Identical effect to render
		// Abstraction of mesh's vertex information on GPU
		// Should be use with std::shared_ptr
		// Holding the information of 
		// Vertex Buffer, Index Buffer, Input Layout 
		// The abstraction for IA and Draw commands
		struct MeshBuffer
		{
		public:
			typedef std::vector<std::unique_ptr<MeshBuffer>> Collection;

			uint32_t                                                IndexCount;
			uint32_t												VertexCount;
			uint32_t                                                StartIndex;
			uint32_t                                                VertexOffset;
			uint32_t                                                VertexStride;
			D3D_PRIMITIVE_TOPOLOGY                                  PrimitiveType;
			DXGI_FORMAT                                             IndexFormat;
			UINT													InputElementCount; // Vertex Description info
			const D3D11_INPUT_ELEMENT_DESC*							pInputElements; // Vertex Description info

			ComPtr<ID3D11InputLayout>               pInputLayout;
			ComPtr<ID3D11Buffer>                    pIndexBuffer;
			ComPtr<ID3D11Buffer>                    pVertexBuffer;

		public:
			~MeshBuffer() {}

			static ComPtr<ID3D11InputLayout>& LookupInputLayout(const D3D11_INPUT_ELEMENT_DESC* pInputElements, IEffect * pEffect);

			VertexDescription GetVertexDescription() const { return VertexDescription(InputElementCount, pInputElements); }

			template<class _TVertex>
			void CreateDeviceResources(ID3D11Device* pDevice, const _TVertex* vertices, unsigned int VerticesCount,IEffect *pEffect = nullptr, D3D_PRIMITIVE_TOPOLOGY primitiveTopology = D3D_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, UINT VertexStride = sizeof(_TVertex), UINT startIndex = 0, UINT VertexOffset = 0);

			template<class _TVertex, class _TIndex>
			void CreateDeviceResources(ID3D11Device* pDevice, const _TVertex* vertices, unsigned int VerticesCount, const _TIndex* indices, unsigned int IndicesCount, IEffect *pEffect = nullptr, D3D_PRIMITIVE_TOPOLOGY primitiveTopology = D3D_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, UINT VertexStride = sizeof(_TVertex), UINT startIndex = 0, UINT VertexOffset = 0);

			void CreateInputLayout(ID3D11Device* pDevice, IEffect *pEffect);
			void CreateInputLayout(ID3D11Device* pDevice, const void * pShaderByteCode, size_t pByteCodeLength);

			bool Empty() const { return VertexCount == 0; }

			template<class TVertex>
			void SetInputElementDescription()
			{
				static_assert(TVertex::InputElementCount, "Valiad Vertex Type should have static InputElements/InputElementCount member");
				InputElementCount = TVertex::InputElementCount;
				pInputElements = TVertex::InputElements;
				VertexStride = sizeof(TVertex);
			}

		public:
			// Setup the Vertex/Index Buffer and call the draw command
			void Draw(ID3D11DeviceContext *pContext, IEffect *pEffect = nullptr) const;
		};

		// Strong typed mesh buffer with static vertex typeing
		template<class _TVertex, class _TIndex>
		struct TypedMeshBuffer : public MeshBuffer
		{
		public:
			static_assert(std::is_integral<_TIndex>::value, "IndexType is not integer");
			static_assert(_TVertex::InputElementCount > 0, "Vertex type not fit the concept");

			typedef TypedMeshBuffer	SelfType;
			typedef _TVertex	VertexType;
			typedef _TIndex		IndexType;
		};

		namespace GeometricPrimtives
		{
			typedef TypedMeshBuffer<VertexPositionNormalTexture, uint16_t> MeshBufferType;
			std::shared_ptr<MeshBufferType>	CreateCube(ID3D11Device * pDevice, float size, bool rhcoords = true);
			std::shared_ptr<MeshBufferType>	CreateSphere(ID3D11Device * pDevice, float radius, size_t tessellation = 16, bool rhcoords = true, bool inside_facing = false);
			std::shared_ptr<MeshBufferType>	CreateCylinder(ID3D11Device * pDevice, float radius, float height, size_t tessellation = 32, bool rhcoords = true);
			std::shared_ptr<MeshBufferType>	CreateCone(ID3D11Device * pDevice, float radius, size_t tessellation = 32, bool rhcoords = true);
		}

		struct DynamicMeshBuffer : public MeshBuffer
		{
		public:
			uint32_t                                                VertexBufferCapacity;
			uint32_t                                                IndexBufferCapacity;

		public:

			template<typename VertexType>
			inline std::enable_if_t<!std::is_void<VertexType>::value> UpdateVertexBuffer(ID3D11DeviceContext* pContext, VertexType * pVertics, size_t verticesCount)
			{
				assert(VertexType::InputElementCount == this->InputElementCount && VertexType::InputElements == this->pInputElements);
				UpdateVertexBuffer(pContext, pVertics, verticesCount, sizeof(VertexType));
			}

			template<typename IndexType>
			inline std::enable_if_t<!std::is_void<IndexType>::value> UpdateIndexBuffer(ID3D11DeviceContext* pContext, IndexType * pIndices, size_t indicesCount)
			{
				assert(ExtractDXGIFormat<IndexType>::value == IndexFormat);
				UpdateIndexBuffer(pContext, pIndices, indicesCount, sizeof(IndexType));
			}

			void UpdateVertexBuffer(ID3D11DeviceContext* pContext, void * pVertics, size_t verticesCount, size_t vertexSize);

			void UpdateIndexBuffer(ID3D11DeviceContext* pContext, void * pIndices, size_t indicesCount, size_t indexSize);

			template<class _TVertex>
			void CreateDeviceResources(ID3D11Device* pDevice, unsigned int VerticesCapacity, IEffect *pEffect = nullptr, D3D_PRIMITIVE_TOPOLOGY primitiveTopology = D3D_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			template<class _TVertex, class _TIndex>
			void CreateDeviceResources(ID3D11Device* pDevice, unsigned int VerticesCapacity, unsigned int IndicesCapacity, IEffect *pEffect = nullptr, D3D_PRIMITIVE_TOPOLOGY primitiveTopology = D3D_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		};

		template<class _TVertex, class _TIndex>
		struct TypedDynamicMeshBuffer : public DynamicMeshBuffer
		{
		public:
		public:
			static_assert(std::is_integral<_TIndex>::value, "IndexType is not integer");
			static_assert(_TVertex::InputElementCount > 0, "Vertex type not fit the concept");

			typedef TypedMeshBuffer	SelfType;
			typedef _TVertex	VertexType;
			typedef _TIndex		IndexType;

			typedef std::vector<VertexType> VertexCollectionType;
			typedef std::vector<IndexType>	IndexCollectionType;

			TypedDynamicMeshBuffer(size_t vertexBufferCapacity, size_t indexBufferCapacity)
			{
				Vertices.reserve(vertexBufferCapacity);
				Indices.reserve(indexBufferCapacity);
			}

			void UpdateVertexBuffer(ID3D11DeviceContext* pContext)
			{
				UpdateVertexBuffer<VertexType>(pContext, Vertices.data(), Vertices.size());
			}

			void UpdateIndexBuffer(ID3D11DeviceContext* pContext)
			{
				UpdateIndexBuffer<IndexType>(pContext, Indices.data(), Indices.size());
			}

			void UpdateMeshBuffer(ID3D11DeviceContext* pContext)
			{
				UpdateVertexBuffer(pContext);
				UpdateIndexBuffer(pContext);
			}

			VertexCollectionType Vertices;
			IndexCollectionType	 Indices;
		};

		//class IModel
		//{
		//public:
		//	virtual void Render(ID3D11DeviceContext *pContext, IEffect* pEffect) = 0;
		//};

		// A ModelPart is a aggregate of a mesh and it's Material
		struct ModelPart
		{
		public:
			std::string						Name;
			std::shared_ptr<MeshBuffer>		pMesh;		// Mesh of this model part
			std::shared_ptr<IMaterial>		pMaterial;	// Material of this model part
			std::shared_ptr<IEffect>		pEffect;	// Default Effect of this model part
			DirectX::BoundingBox			BoundBox;
			DirectX::BoundingOrientedBox	BoundOrientedBox;

			// This render method will not set "Transform"
			void Render(ID3D11DeviceContext *pContext, IEffect* pEffect = nullptr);
		};

		class LocalMatrixHolder : virtual public ILocalMatrix
		{
		public:
			// Inherited via ILocalMatrix
			void XM_CALLCONV SetModelMatrix(DirectX::FXMMATRIX model)
			{
				XMStoreFloat4x4(&LocalMatrix, model);
			}

			XMMATRIX GetModelMatrix() const
			{
				return XMLoadFloat4x4(&LocalMatrix);
			}
			Matrix4x4	LocalMatrix;
		};

		class IModelNode : virtual public IBoundable
		{
		public:
			virtual ~IModelNode();

			// Property : Name
			virtual const std::string& Name() const = 0;
			virtual void SetName(const std::string& name) = 0;

			// Method : Render
			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, IEffect* pEffect = nullptr) = 0;

			// Get the Recursive multiplied model matrix since root, return the global world matrix
			// virtual XMMATRIX GetWorldMatrix() const;
			//virtual XMMATRIX GetModelMatrix() const override;
			// Transformed OrientedBounding Box
			//virtual BoundingOrientedBox GetOrientedBoundingBox() const = 0;
			// Transformed Bounding Box
			//virtual BoundingBox GetBoundingBox() const override;
			// Transformed Bounding Sphere
			// virtual BoundingSphere GetBoundingSphere() const;

			//virtual void XM_CALLCONV SetModelMatrix(DirectX::FXMMATRIX model) override;

			//virtual XMMATRIX GetModelMatrix() const override;


			// Inherited via ILocalMatrix
			//virtual void XM_CALLCONV SetModelMatrix(DirectX::FXMMATRIX model) override;
			//virtual XMMATRIX GetModelMatrix() const override;

#pragma region PropertyParent
//	IModelNode* Parent() { return pParent; }
//	const IModelNode* Parent() const { return pParent; }
//	void SetParent(IModelNode* parent) { pParent = parent; }

//private:
//	IModelNode*			pParent = nullptr;
#pragma endregion

//BoundingOrientedBox BoundOrientedBox;
//BoundingSphere		BoundSphere;
//BoundingBox			BoundBox;
//Matrix4x4			LocalMatrix;
//float				Opticity;
		};

		// A Monolith Model is a model with single ModelPart
		class MonolithModel : public IModelNode, public ModelPart
		{
		public:
			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, IEffect* pEffect) override;

			const std::string& Name() const override { return ModelPart::Name; }
			void SetName(const std::string& name) { ModelPart::Name = name; }

			// Inherited via IModelNode
			virtual BoundingBox GetBoundingBox() const override;
			virtual BoundingOrientedBox GetOrientedBoundingBox() const override;

		};

		// A basic model is a collection of ModelPart shares same Local Matrix
		// Leaf node in a model tree
		class CompositionModel : public IModelNode
		{
		protected:
			std::string				_Name;
		public:
			BoundingOrientedBox		BoundOrientedBox;
			BoundingSphere			BoundSphere;
			BoundingBox				BoundBox;

			std::vector<ModelPart>	Parts;

			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, IEffect* pEffect) override;
			const std::string& Name() const override { return _Name; }
			void SetName(const std::string& name) { _Name = name; }

			// Inherited via IModelNode
			virtual BoundingBox GetBoundingBox() const override;
			virtual BoundingOrientedBox GetOrientedBoundingBox() const override;
		};

		// Inherit from vector type, only push_back is valiad !!!
		class CollectionModel : public IModelNode
		{
		public:
			struct ModelTransformPair
			{
				LinearTransform	Transform;
				std::shared_ptr<IModelNode> Model;
			};

			typedef std::vector<ModelTransformPair> ContainnerType;

		protected:
			std::string				_Name;
		public:
			ContainnerType			Children;
		public:
			//BoundingOrientedBox		BoundOrientedBox;
			BoundingBox				BoundBox;

		public:

			// All the data in Children's postion/orientation is 
			// In the local coordinate of it's parent!
			void AddChild(const std::shared_ptr<IModelNode> &model, const LinearTransform &transform = reinterpret_cast<const LinearTransform&>(LinearTransform::Identity));
			//void push_back(const value_type& _Val);
			//void push_back(value_type&& _Val);
			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, IEffect* pEffect) override;

			const std::string& Name() const { return _Name; }
			void SetName(const std::string& name) { _Name = name; }

			// Inherited via IModelNode
			virtual BoundingBox GetBoundingBox() const override;
			//virtual BoundingOrientedBox GetOrientedBoundingBox() const override;

		};

		class IDynamicAsset
		{
		public:
			virtual ~IDynamicAsset();
			// Check if the GPU buffer is flaged as DYNAMIC
			virtual bool IsDeviceResourceDynamic() const;;

			// Check if the data have been loaded to CPU memery
			virtual bool IsInMemery() const;

			virtual bool IsInDevice() const;

			// Create GPU objects using Vertex/Index Buffer in CPU
			virtual bool CreateDeviceResource(ID3D11Device *pDevice) = 0;

			// Update GPU buffers using Vertex/Index Buffer in CPU
			virtual bool UpdateDeviceResource(ID3D11DeviceContext *pContext);

			// Release the Vertex/Index Buffer in CPU memery
			virtual void ReleaseDynamicResource();

			// To a amx binary stream , default to false
			virtual bool Serialize(std::ostream& binary) const;

			// from a amx binary stream
			virtual bool Deserialize(std::istream& binary);

			// Reload from file to CPU
			virtual bool Reload() = 0;
		};

		// This Model also keeps the geomreics data in CPU
		class DefaultStaticModel : public CompositionModel, public IDynamicAsset
		{
		public:
			typedef VertexPositionNormalTexture				VertexType;
			typedef uint16_t								IndexType;
			typedef FacetPrimitives::Triangle<IndexType>	TriangleType;

			static DefaultStaticModel * CreateFromObjFile(const std::wstring &file, ID3D11Device * pDevice = nullptr, const std::wstring& textureDir = L"", bool flipNormal = false);
			static DefaultStaticModel * CreateFromFbxFile(const std::wstring &file, ID3D11Device * pDevice = nullptr, const std::wstring& textureDir = L"");
			static DefaultStaticModel * CreateFromSmxFile(ID3D11Device *pDevice, const std::wstring &file);

			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, IEffect* pEffect = nullptr) override;

			// Check if the data have been loaded to CPU memery
			bool IsInMemery() const override;
			// Create GPU objects using Vertex/Index Buffer in CPU
			bool CreateDeviceResource(ID3D11Device *pDevice) override;
			// Release the Vertex/Index Buffer in CPU memery
			void ReleaseDynamicResource() override;

			// To a obj text stream
			bool Serialize(std::ostream& binary) const override;
			// from a obj text stream
			bool Deserialize(std::istream& binary) override;

			// Reload from file to CPU
			bool Reload() override;

			DefaultStaticModel();
			~DefaultStaticModel();
		public:
			std::vector<VertexType>								Vertices;
			std::vector<TriangleType>							Facets;
			stdx::stride_range<Vector3>							Positions;
			stdx::stride_range<Vector3>							Normals;
			stdx::stride_range<Vector2>							TexCoords;

		private:
			bool				m_IsLoaded;
			const std::wstring  m_FilePath;
			const std::wstring  m_TexDir;
		};

		struct SkinMeshData;

		class ISkinningModel
		{
		public:
			virtual size_t GetBonesCount() const = 0;
			//virtual XMMATRIX* GetBoneTransforms() = 0;
			//inline const XMMATRIX* GetBoneTransforms() const
			//{
			//	return const_cast<ISkinningModel*>(this)->GetBoneTransforms();
			//}
			virtual const BoundingOrientedBox* GetBoneBoundingBoxes() const = 0;
			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, const XMMATRIX* boneTransforms , IEffect* pEffect = nullptr) = 0;
		};

		class DefaultSkinningModel :
			public CompositionModel, public IDynamicAsset , public ISkinningModel
		{
		public:
			typedef VertexPositionNormalTangentColorTextureSkinning VertexType;
			typedef uint16_t IndexType;
			typedef FacetPrimitives::Triangle<IndexType> TriangleType;

			static DefaultSkinningModel* CreateFromFbxFile(const std::string& file, const std::wstring& textureDir = L"", ID3D11Device* pDevice = nullptr);
			static DefaultSkinningModel* CreateFromAmxFile(const std::string& file, ID3D11Device* pDevice = nullptr);
			static DefaultSkinningModel* CreateFromData(SkinMeshData* pData, const std::wstring& textureDir = L"", ID3D11Device* pDevice = nullptr);
			static DefaultSkinningModel* CreateFromDatas(std::list<SkinMeshData>& datas, const std::wstring& textureDir = L"", ID3D11Device* pDevice = nullptr);

			// Check if the data have been loaded to CPU memery
			bool IsInMemery() const override;
			// Create GPU objects using Vertex/Index Buffer in CPU
			bool CreateDeviceResource(ID3D11Device *pDevice) override;
			// Release the Vertex/Index Buffer in CPU memery
			void ReleaseDynamicResource();

			// To a amx binary stream
			bool Serialize(std::ostream& binary) const override;
			// from a amx binary stream
			bool Deserialize(std::istream& binary) override;
			// Reload from file to CPU
			bool Reload() override;

			// IModelNode
			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, IEffect* pEffect = nullptr) override;
			virtual void Render(ID3D11DeviceContext *pContext, const Matrix4x4& transform, const XMMATRIX* boneTransforms , IEffect* pEffect = nullptr) override;

			// Inherited via ISkinningModel
			virtual size_t GetBonesCount() const override;
			//virtual DirectX::XMMATRIX* GetBoneTransforms() override;
			virtual const BoundingOrientedBox* GetBoneBoundingBoxes() const override;

			DefaultSkinningModel();
			~DefaultSkinningModel();

		public:
			//std::vector<Matrix4x4,
			//	AlignedAllocator<Matrix4x4, 16U >>
			//	BoneTransforms;

		public:
			stdx::stride_range<VertexType>							Vertices;
			stdx::stride_range<TriangleType>						Facets;
			stdx::stride_range<Vector3>								Positions;
			stdx::stride_range<Vector3>								Normals;
			stdx::stride_range<Vector2>								TexCoords;
			stdx::stride_range<Vector4>								Tagents;
			stdx::stride_range<uint32_t>							BlendWeights;
			stdx::stride_range<uint32_t>							BlendIndices;

		private:
			bool				m_IsLoaded;
			const std::string	m_FilePath;


			uint32_t	m_VertexCount;
			uint32_t	m_IndexCount;
			uint32_t	m_BonesCount;

			std::unique_ptr<VertexType[]>	m_Vertices;
			std::unique_ptr<IndexType[]>	m_Indices;
			std::unique_ptr<Matrix4x4[]>	m_DefaultBoneTransforms;

			std::vector<BoundingOrientedBox> m_BoneBoxes;

		protected:

			void SetFromSkinMeshData(std::list<SkinMeshData> &meshes, const std::wstring& textureDir = L"");
			void SetFromSkinMeshData(SkinMeshData* pData, const std::wstring& textureDir = L"");
			void ResetRanges();
			void CaculateBoneBoxes(const Matrix4x4* defaultBoneTransforms);
		};

		template <class _TVertex>
		inline void MeshBuffer::CreateDeviceResources(ID3D11Device * pDevice, const _TVertex * vertices, unsigned int VerticesCount, IEffect * pEffect, D3D_PRIMITIVE_TOPOLOGY primitiveTopology, UINT vertexStride, UINT startIndex, UINT vertexOffset)
		{
			CreateDeviceResources<_TVertex, int>(pDevice, vertices, VerticesCount, nullptr, 0, pEffect, primitiveTopology, vertexStride, startIndex, vertexOffset);
		}

		template <class _TVertex, class _TIndex>
		inline void MeshBuffer::CreateDeviceResources(ID3D11Device * pDevice, const _TVertex * vertices, unsigned int VerticesCount, const _TIndex * indices, unsigned int IndicesCount, IEffect * pEffect, D3D_PRIMITIVE_TOPOLOGY primitiveTopology, UINT vertexStride, UINT startIndex, UINT vertexOffset)
		{
			SetInputElementDescription<_TVertex>();

			assert(pDevice != nullptr);

			if (pEffect != nullptr)
			{
				CreateInputLayout(pDevice, pEffect);
			}


			this->pVertexBuffer = DirectX::CreateVertexBuffer<_TVertex>(pDevice, VerticesCount, vertices);
			if (indices != nullptr && IndicesCount > 0)
				this->pIndexBuffer = DirectX::CreateIndexBuffer(pDevice, IndicesCount, indices);
			this->IndexFormat = ExtractDXGIFormat<_TIndex>::value;
			this->VertexCount = VerticesCount;
			this->IndexCount = IndicesCount;
			this->StartIndex = startIndex;
			this->VertexStride = vertexStride;
			this->VertexOffset = vertexOffset;
			this->PrimitiveType = primitiveTopology;
		}

		template<class _TVertex>
		inline void DynamicMeshBuffer::CreateDeviceResources(ID3D11Device * pDevice, unsigned int VerticesCapacity, IEffect * pEffect, D3D_PRIMITIVE_TOPOLOGY primitiveTopology)
		{
			CreateDeviceResources<_TVertex, void>(pDevice, VerticesCapacity, 0, pEffect, primitiveTopology);
		}

		template<class _TVertex, class _TIndex>
		inline void DynamicMeshBuffer::CreateDeviceResources(ID3D11Device * pDevice, unsigned int VerticesCapacity, unsigned int IndicesCapacity, IEffect * pEffect, D3D_PRIMITIVE_TOPOLOGY primitiveTopology)
		{
			SetInputElementDescription<_TVertex>();

			assert(pDevice != nullptr);

			if (pEffect != nullptr)
				CreateInputLayout(pDevice, pEffect);

			this->VertexBufferCapacity = VerticesCapacity;
			this->pVertexBuffer = DirectX::CreateVertexBuffer<_TVertex>(pDevice, this->VertexBufferCapacity, nullptr, D3D11_CPU_ACCESS_WRITE);

			this->IndexFormat = ExtractDXGIFormat<_TIndex>::value;;
			this->IndexBufferCapacity = this->IndexFormat != DXGI_FORMAT_UNKNOWN ? IndicesCapacity : 0;

			if (this->IndexBufferCapacity > 0)
				this->pIndexBuffer = DirectX::CreateIndexBuffer<_TVertex>(pDevice, this->IndexBufferCapacity, nullptr, D3D11_CPU_ACCESS_WRITE);

			this->VertexCount = 0;
			this->IndexCount = 0;
			this->StartIndex = 0;
			this->VertexStride = sizeof(_TVertex);
			this->VertexOffset = 0;
			this->PrimitiveType = primitiveTopology;
		}
}
}

