#include "pch_directX.h"
#include <fstream>
#include <WICTextureLoader.h>
#include <DDSTextureLoader.h>
#include "Material.h"
#include "tiny_obj_loader.h"
#include <ShaderEffect.h>
#include <ShadowMapGenerationEffect.h>
#include <filesystem>

using namespace std;
using namespace std::tr2::sys;
using namespace DirectX::Scene;
using namespace DirectX;
using namespace tinyobj;
using namespace Microsoft::WRL;

PhongMaterial::PhongMaterial()
	: pDefaultRequestEffect(nullptr)
{
}

PhongMaterial::PhongMaterial(const PhongMaterialData & data, const std::wstring & lookupDirectory, ID3D11Device * pDevice)
	: PhongMaterialData(data) , pDefaultRequestEffect(nullptr)
{
	path lookup(lookupDirectory);
	if (!AmbientMapName.empty())
		AmbientMapName = (lookup / AmbientMapName).string();
	if (!DiffuseMapName.empty())
		DiffuseMapName = (lookup / DiffuseMapName).string();
	if (!SpecularMapName.empty())
		SpecularMapName = (lookup / SpecularMapName).string();
	if (!NormalMapName.empty())
		NormalMapName = (lookup / NormalMapName).string();
	if (!DisplaceMapName.empty())
		DisplaceMapName = (lookup / DisplaceMapName).string();
	if (!EmissiveMapName.empty())
		EmissiveMapName = (lookup / EmissiveMapName).string();
	CreateDeviceResources(pDevice);
}

PhongMaterial::~PhongMaterial()
{
}

std::vector<std::shared_ptr<PhongMaterial>> PhongMaterial::CreateFromMtlFile(ID3D11Device * pDevice, const std::wstring & file, const std::wstring & lookupDirectory)
{
	std::vector<std::shared_ptr<PhongMaterial>> Materials;
	std::map<std::string, int> map;
	std::vector<material_t> materis;
	std::ifstream fin(file);
	tinyobj::LoadMtl(map, materis, fin);
	path lookup(lookupDirectory);

	Materials.reserve(materis.size());

	for (auto& mat : materis)
	{
		auto pMaterial = make_shared<PhongMaterial>();
		pMaterial->Name = mat.name;
		pMaterial->Alpha = mat.dissolve;
		pMaterial->DiffuseColor = Color(mat.diffuse);
		pMaterial->AmbientColor = Color(mat.ambient);
		pMaterial->SpecularColor = Color(mat.specular);
		if (!mat.diffuse_texname.empty())
		{
			auto fileName = lookup / mat.diffuse_texname;
			pMaterial->DiffuseMapName = fileName.string();
		}
		if (!mat.specular_texname.empty())
		{
			auto fileName = lookup / mat.specular_texname;
			pMaterial->SpecularMapName = fileName.string();
		}
		if (!mat.normal_texname.empty())
		{
			auto fileName = lookup / mat.normal_texname;
			pMaterial->NormalMapName = fileName.string();
		}
		if (!mat.ambient_texname.empty())
		{
			auto fileName = lookup / mat.normal_texname;
			pMaterial->AmbientMapName = fileName.string();
		}
		Materials.push_back(pMaterial);
	}

	for (auto& pMat : Materials)
	{
		pMat->CreateDeviceResources(pDevice);
	}
	return Materials;
}

std::shared_ptr<PhongMaterial> PhongMaterial::CreateFromMaterialData(const PhongMaterialData& data, const std::wstring & lookupDirectory, ID3D11Device * pDevice)
{
	return std::make_shared<PhongMaterial>(data, lookupDirectory, pDevice);
}

void tolower(std::string& str)
{
	for (auto& c : str)
	{
		if (c >= 'A' && c <= 'Z')
			c += 'a' - 'A';
	}
}

HRESULT CreateTextureFromFile(ID3D11Device* d3dDevice,
	const path& fileName,
	ID3D11Resource** texture,
	ID3D11ShaderResourceView** textureView,
	size_t maxsize = 0)
{
	auto ext = fileName.extension().string();
	tolower(ext);
	if (ext == ".dds")
		return CreateDDSTextureFromFile(d3dDevice, fileName.wstring().data(), texture, textureView, maxsize);
	else
	{
		return CreateWICTextureFromFile(d3dDevice, fileName.wstring().data(), texture, textureView, maxsize);
	}
}

void PhongMaterial::CreateDeviceResources(ID3D11Device * pDevice, bool forceUpdate)
{
	AmbientMap.Reset();
	DiffuseMap.Reset();
	SpecularMap.Reset();
	EmissiveMap.Reset();
	NormalMap.Reset();
	DisplaceMap.Reset();

	if (!pDevice) return;

	ComPtr<ID3D11DeviceContext> pContext;
	ComPtr<ID3D11Resource> pResource;
	pDevice->GetImmediateContext(&pContext);
	path fileName;

	if (!DiffuseMapName.empty())
	{
		fileName = DiffuseMapName;
		CreateTextureFromFile(pDevice, fileName.wstring().data(), &pResource, &DiffuseMap);
	}
	if (!SpecularMapName.empty())
	{
		fileName = SpecularMapName;
		CreateTextureFromFile(pDevice, fileName.wstring().data(), &pResource, &SpecularMap);
	}
	if (!NormalMapName.empty())
	{
		fileName = NormalMapName;
		CreateTextureFromFile(pDevice, fileName.wstring().data(), &pResource, &NormalMap);
	}
	if (!AmbientMapName.empty())
	{
		fileName = AmbientMapName;
		CreateTextureFromFile(pDevice, fileName.wstring().data(), &pResource, &AmbientMap);
	}
	if (!DisplaceMapName.empty())
	{
		fileName = DisplaceMapName;
		CreateTextureFromFile(pDevice, fileName.wstring().data(), &pResource, &DisplaceMap);
	}
}

void PhongMaterial::SetupEffect(IEffect *pEffect) const
{
	if (pEffect == nullptr)
		pEffect = pDefaultRequestEffect;

	if (pEffect == nullptr)
		return;

	auto pPhongEffect = dynamic_cast<IEffectPhongMaterial*>(pEffect);
	if (pPhongEffect)
	{
		pPhongEffect->SetDiffuseColor(DiffuseColor);
		pPhongEffect->SetEmissiveColor(EmissiveColor);
		pPhongEffect->SetSpecularColor(SpecularColor);
		pPhongEffect->SetSpecularPower(SpecularColor.A());
		pPhongEffect->SetAlpha(GetAlpha());
		pPhongEffect->SetDiffuseMap(GetDiffuseMap());
		pPhongEffect->SetNormalMap(GetNormalMap());
		pPhongEffect->SetSpecularMap(GetSpecularMap());
		pPhongEffect->SetAlphaDiscard(UseAlphaDiscard);
		return;
	}

	auto pSGEffect = dynamic_cast<ShadowMapGenerationEffect*>(pEffect);
	if (pSGEffect && pSGEffect->GetShadowFillMode() != ShadowMapGenerationEffect::BoneColorFill)
	{
		if (DiffuseMap != nullptr && UseAlphaDiscard)
		{
			pSGEffect->SetAlphaDiscardTexture(DiffuseMap.Get());
			pSGEffect->SetAlphaDiscardThreshold(0.15f);
		}
		else
		{
			pSGEffect->DisableAlphaDiscard();
		}
		return;
	}


	auto pMEffect = dynamic_cast<BasicEffect*>(pEffect);

	if (pMEffect)
	{
		pMEffect->SetAlpha(GetAlpha());
		pMEffect->SetDiffuseColor(DiffuseColor);
		pMEffect->SetEmissiveColor(EmissiveColor);
		pMEffect->SetSpecularColor(SpecularColor);
		pMEffect->SetSpecularPower(1.0f);


		if (GetDiffuseMap())
		{
			pMEffect->SetTextureEnabled(true);
			pMEffect->SetTexture(GetDiffuseMap());

		}
		else
		{
			pMEffect->SetTextureEnabled(false);
		}
		return;
	}

	auto pSEffect = dynamic_cast<SkinnedEffect*>(pEffect);
	if (pSEffect)
	{
		pSEffect->SetAlpha(GetAlpha());
		pSEffect->SetTexture(GetDiffuseMap());
		pSEffect->SetDiffuseColor(DiffuseColor);
		pSEffect->SetEmissiveColor(EmissiveColor);
		pSEffect->SetSpecularColor(SpecularColor);
		pSEffect->SetSpecularPower(1.0f);
		return;
	}

	auto pDGSLEffect = dynamic_cast<DGSLEffect*>(pEffect);
	if (pDGSLEffect)
	{
		pDGSLEffect->SetAlpha(GetAlpha());
		pDGSLEffect->SetDiffuseColor(DiffuseColor);
		pDGSLEffect->SetAmbientColor(AmbientColor);
		pDGSLEffect->SetEmissiveColor(EmissiveColor);
		pDGSLEffect->SetSpecularColor(SpecularColor);
		pDGSLEffect->SetSpecularPower(1.0f);

		if (GetDiffuseMap())
		{
			pDGSLEffect->SetTextureEnabled(true);
			pDGSLEffect->SetTexture(GetDiffuseMap());
		}
		else
		{
			pDGSLEffect->SetTextureEnabled(false);
		}
		return;
	}

}


// Inherited via IMaterial

Color PhongMaterial::GetAmbientColor() const
{
	return AmbientColor;
}

Color PhongMaterial::GetDiffuseColor() const
{
	return DiffuseColor;
}

Color PhongMaterial::GetSpecularColor() const
{
	return SpecularColor;
}

float PhongMaterial::GetAlpha() const
{
	return Alpha;
}

bool PhongMaterial::GetUseAlphaDiscard() const
{
	return UseAlphaDiscard;
}

ID3D11ShaderResourceView * PhongMaterial::GetDiffuseMap() const
{
	return DiffuseMap.Get();
}

ID3D11ShaderResourceView * PhongMaterial::GetNormalMap() const
{
	return NormalMap.Get();
}

ID3D11ShaderResourceView * PhongMaterial::GetSpecularMap() const
{
	return SpecularMap.Get();
}

ID3D11ShaderResourceView * PhongMaterial::GetDisplaceMap() const
{
	return DisplaceMap.Get();
}

void DirectX::Scene::IMaterial::CreateDeviceResources(ID3D11Device * pDevice, bool forceUpdate)
{}

DirectX::Scene::IMaterial::~IMaterial() {}
