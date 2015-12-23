#pragma once
//#include <wrl\client.h>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID2D1Factory;
struct IDWriteFactory;
struct ID2D1DeviceContext;
struct ID2D1Device;
struct ID2D1Bitmap;

namespace DirectX
{
	class IPostEffect;
	class IEffect;
	class IEffectFactory;

	class Texture;
	class Texture2D;
	class DynamicTexture2D;
	class RenderableTexture2D;
	class StagingTexture2D;
	class DepthStencilBuffer;
	class DoubleBufferTexture2D;
	class RenderTarget;
	class EnvironmentTexture;

	class ILocatable;
	class IOriented;
	class IScalable;
	class IBoundable;
	class IRigid;
	class ILocalMatrix;

	class DeviceResources;

	namespace Scene
	{
		struct MeshBuffer;
		struct DynamicMeshBuffer;
		struct ModelPart;
		struct SkinMeshData;

		class IModelNode;
		class ISkinningModel;
		class IDynamicAsset;

		class IMaterial;
		class PhongMaterial;

		struct MaterialData;
		struct LambertMaterialData;
		struct PhongMaterialData;

		class HUDElement;
		class HUDCanvas;
		class Panel;
		class TextBlock;
	}
}

namespace Causality
{
	using DirectX::IPostEffect;
	using DirectX::IEffect;
	using DirectX::IEffectFactory;

	using DirectX::Texture;
	using DirectX::Texture2D;
	using DirectX::DynamicTexture2D;
	using DirectX::RenderableTexture2D;
	using DirectX::StagingTexture2D;
	using DirectX::DepthStencilBuffer;
	using DirectX::DoubleBufferTexture2D;
	using DirectX::RenderTarget;
	using DirectX::EnvironmentTexture;

	using DirectX::IRigid;

	using DirectX::Scene::MeshBuffer;
	using DirectX::Scene::DynamicMeshBuffer;
	using DirectX::Scene::IModelNode;
	using DirectX::Scene::ISkinningModel;
	using DirectX::Scene::IDynamicAsset;
	using DirectX::Scene::IMaterial;

	using DirectX::Scene::PhongMaterial;
	using DirectX::Scene::MaterialData;
	using DirectX::Scene::LambertMaterialData;
	using DirectX::Scene::PhongMaterialData;

	using DirectX::Scene::HUDElement;
	using DirectX::Scene::HUDCanvas;
	using DirectX::Scene::Panel;
	using DirectX::Scene::TextBlock;


	using DirectX::DeviceResources;

	using IRenderDevice  = ID3D11Device;
	using IRenderContext = ID3D11DeviceContext;
	using I2DFactory     = ID2D1Factory;
	using I2DContext     = ID2D1DeviceContext;
	using ITextFactory   = IDWriteFactory;

	//class RenderDevice : public Microsoft::WRL::ComPtr<ID3D11Device>
	//{
	//public:
	//	typedef Microsoft::WRL::ComPtr <ID3D11Device> base_type;
	//	using base_type::operator->;
	//	using base_type::operator&;
	//	using base_type::operator=;
	//	using base_type::operator Microsoft::WRL::Details::BoolType;

	//	RenderDevice() = default;
	//	RenderDevice(ID3D11Device* pDevice)
	//		: base_type(pDevice)
	//	{
	//	}

	//	operator ID3D11Device*()
	//	{
	//		return Get();
	//	}
	//};

	//class IRenderContext : public Microsoft::WRL::ComPtr<ID3D11DeviceContext>
	//{
	//public:
	//	typedef Microsoft::WRL::ComPtr <ID3D11DeviceContext> base_type;
	//	using base_type::operator->;
	//	using base_type::operator&;
	//	using base_type::operator=;
	//	using base_type::operator Microsoft::WRL::Details::BoolType;

	//	IRenderContext() = default;
	//	IRenderContext(ID3D11DeviceContext* pContext)
	//		: base_type(pContext)
	//	{}

	//	operator ID3D11DeviceContext*()
	//	{
	//		return Get();
	//	}
	//};

}