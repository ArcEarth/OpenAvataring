#include "pch_bcl.h"
#include "Models.h"
#include "FbxParser.h"
#include <Effects.h>
using namespace DirectX;
using namespace DirectX::Scene;
using namespace Causality;

DefaultSkinningModel* DefaultSkinningModel::CreateFromFbxFile(const std::string& file, const std::wstring& textureDir, ID3D11Device* pDevice)
{
	FbxParser fbx;
	fbx.ImportMesh(file);
	auto meshes = fbx.GetMeshs();
	if (meshes.empty())
		return nullptr;
	auto pModel = new DefaultSkinningModel();
	pModel->SetFromSkinMeshData(&meshes.front(), textureDir);
	if (pDevice)
	{
		pModel->CreateDeviceResource(pDevice);
	}
	return pModel;
}