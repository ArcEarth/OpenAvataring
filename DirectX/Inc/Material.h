#include "DirectXMathExtend.h"
#include <d3d11.h>
#include <vector>
#include <wrl\client.h>
#include <memory>
#include "Textures.h"
#include "MaterialData.h"

namespace DirectX
{
	namespace Scene
	{
		// Abstraction for the requirement about Pixel shader process
		class IMaterial abstract
		{
		public:
			virtual void SetupEffect(IEffect *pEffect) const = 0;
			virtual void CreateDeviceResources(ID3D11Device* pDevice, bool forceUpdate = false);;

			virtual ~IMaterial();
			//virtual Color	GetColor(const char* key) const = 0;
			//virtual float	GetFloat(const char* key) const = 0;
			//virtual int		GetInt(const char* key) const = 0;
			//virtual int		GetString(const char* key) const = 0;
			//virtual ID3D11ShaderResourceView*	GetTexture(const char* key) const = 0;
			//virtual Color	SetColor(const char* key) const = 0;
			//virtual void SetFloat(const char* key) const = 0;
			//virtual void SetInt(const char* key) const = 0;
			//virtual void SetString(const char* key) const = 0;
			//virtual void SetTexture(const char* key) const = 0;
		};

		//0. Color on and Ambient off
		//1. Color on and Ambient on
		//2. Highlight on
		//3. Reflection on and Ray trace on
		//4. Transparency: Glass on, Reflection : Ray trace on
		//5. Reflection : Fresnel on and Ray trace on
		//6. Transparency : Refraction on, Reflection : Fresnel off and Ray trace on
		//7. Transparency : Refraction on, Reflection : Fresnel on and Ray trace on
		//8. Reflection on and Ray trace off
		//9. Transparency : Glass on, Reflection : Ray trace off
		//10. Casts shadows onto invisible surfaces
		enum ObjMateriaIlluminitionModel
		{
			ColorOnAmbientOff = 0,
			ColorOnAmbientOn = 1,
			HighlightOn = 2,
			ReflectionOnRayTraceOn = 3,
			TransparencyOn = 4,
		};

		//class PropertyMap : private std::map<std::string, void*>
		//{
		//public:
		//	typedef std::map<std::string, void*> base_type;
		//	template <typename T>
		//	T	Get(const std::string& key) const
		//	{
		//		auto itr = find(key);
		//		if (itr != end())
		//			return reinterpret_cast<T>(itr->second);
		//		else
		//			return T();
		//	}

		//	template <typename T>
		//	void Set(const std::string& key, const T& value)
		//	{
		//		base_type::operator[](key) = value;
		//	}

		//	using base_type::operator[];

		//	bool HasProperty(const std::string& key) const
		//	{
		//		return find(key) != end();
		//	}

		//	using base_type::begin;
		//	using base_type::end;
		//	using base_type::size;

		//	size_t Size() const { return size(); }

		//	const std::map<std::string, void*>& Properties() const
		//	{
		//		return *this;
		//	}
		//};

		//class Material : public IMaterial, public PropertyMap
		//{
		//public:
		//	std::string RequstedEffectName;

		//	Color		GetColor(const std::string& key) const
		//	{
		//		return Get<Color>(key);
		//	}
		//	float		GetFloat(const std::string& key) const
		//	{
		//		return Get<float>(key);
		//	}
		//	int			GetInt(const std::string& key) const
		//	{
		//		return Get<int>(key);
		//	}
		//	Vector4		GetVector4(const std::string& key) const
		//	{
		//		return Get<Vector4>(key);
		//	}
		//	std::string	GetString(const std::string& key) const
		//	{
		//		return Get<std::string>(key);
		//	}
		//	const Texture&	GetTexture(const std::string& key) const
		//	{
		//		return *Get<const Texture*>(key);
		//	}

		//	Color		                GetAmbientColor() const { return GetColor("AmbientColor"); }
		//	Color		                GetDiffuseColor() const { return GetColor("DiffuseColor"); }
		//	Color		                GetSpecularColor() const { return GetColor("SpecularColor"); }
		//	float		                GetSpecularPower() const { return GetFloat("SpecularPower"); }
		//	float		                GetOpacity() const { return GetFloat("Opacity"); }
		//	ID3D11ShaderResourceView*	GetDiffuseMap() const { return GetTexture("DiffuseMap"); }
		//	ID3D11ShaderResourceView*	GetNormalMap() const { return GetTexture("NormalMap"); }
		//	ID3D11ShaderResourceView*	GetDisplaceMap() const { return GetTexture("DisplaceMap"); }
		//	ID3D11ShaderResourceView*	GetSpecularMap() const { return GetTexture("SpecularMap"); }

		//	void		SetColor(const std::string& key, const Color& value)
		//	{
		//		Set(key, value);
		//	}
		//	void		SetFloat(const std::string& key, float value)
		//	{
		//		Set(key, value);
		//	}
		//	void		SetInt(const std::string& key, int value)
		//	{
		//		Set(key, value);
		//	}
		//	void		SetString(const std::string& key, const std::string& value)
		//	{
		//		Set(key, value);
		//	}
		//	void		SetTexture(const std::string& key, Texture& texture)
		//	{
		//		Set(key, &texture);
		//	}
		//	void		SetVector4(const std::string& key, const Vector4 value)
		//	{
		//		Set(key, value);
		//	}
		//};

		class PhongMaterial : public IMaterial, public PhongMaterialData
		{
		public:
			using SRVComPtr = Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>;

			PhongMaterial();
			PhongMaterial(const PhongMaterialData& data, const std::wstring &lookupDirectory, ID3D11Device* pDevice = nullptr);
			~PhongMaterial();
			static std::vector<std::shared_ptr<PhongMaterial>> CreateFromMtlFile(ID3D11Device* pDevice, const std::wstring &file, const std::wstring &lookupDirectory);
			static std::shared_ptr<PhongMaterial> CreateFromMaterialData(const PhongMaterialData &data, const std::wstring &lookupDirectory, ID3D11Device* pDevice = nullptr);

			void CreateDeviceResources(ID3D11Device* pDevice, bool forceUpdate = false) override;

			IEffect*	pDefaultRequestEffect;

			SRVComPtr	AmbientMap;
			SRVComPtr	DiffuseMap;
			SRVComPtr	EmissiveMap;
			SRVComPtr	NormalMap;
			SRVComPtr	DisplaceMap;
			SRVComPtr	SpecularMap;

			// Inherited via IMaterial
			virtual void SetupEffect(IEffect *pEffect) const override;

			Color GetAmbientColor() const;
			Color GetDiffuseColor() const;
			Color GetSpecularColor() const;
			float GetAlpha() const;
			bool  GetUseAlphaDiscard() const;
			ID3D11ShaderResourceView * GetDiffuseMap() const;
			ID3D11ShaderResourceView * GetNormalMap() const;
			ID3D11ShaderResourceView * GetSpecularMap() const ;
			ID3D11ShaderResourceView * GetDisplaceMap() const ;
		};
	}
}