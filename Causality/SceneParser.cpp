#include "pch_bcl.h"
#include "Scene.h"
#include "SceneParser.h"
#include <Model.h>
#include <tinyxml2.h>
#include "Settings.h"
#include "AssetDictionary.h"
#include <filesystem>
#include <sstream>

using namespace tinyxml2;
using namespace Causality;
using namespace DirectX::Scene;
using namespace std;
using namespace std::tr2::sys;

void ParseSceneSettings(tinyxml2::XMLElement * nScene);

namespace Causality
{
	extern size_t CLIP_FRAME_COUNT;
}

#pragma region GetAttribute Helper
template <typename T>
void GetAttribute(_In_ XMLElement* node, _In_ const char* attr, _Inout_ T& value)
{
	static_assert(false, "unsupported type T");
}

template <>
void GetAttribute<Vector3>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ Vector3& value)
{
	auto attrval = node->Attribute(attr);
	if (attrval != nullptr)
	{
		string str(attrval);
		if (str.find_first_of(',') != string::npos)
		{
			stringstream ss(str);
			char ch;
			ss >> value.x >> ch >> value.y >> ch >> value.z;
		}
		else
		{
			value.x = value.y = value.z = (float)atof(attrval); // replicate
		}
	}
}

template <>
void GetAttribute<DirectX::Vector4>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ DirectX::Vector4& value)
{
	auto attrval = node->Attribute(attr);
	if (attrval != nullptr)
	{
		stringstream ss(attrval);
		char ch;
		ss >> value.x >> ch >> value.y >> ch >> value.z >> ch >> value.w;
	}
}


template <>
// Value format : "#AARRGGBB"
void GetAttribute<DirectX::Color>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ DirectX::Color& value)
{
	auto attrval = node->Attribute(attr);
	if (attrval != nullptr && attrval[0] == '#')
	{
		char* end;
		auto val = strtoul(attrval + 1, &end, 16);
		value.w = ((float)((val & 0xff000000U) >> 24)) / (float)0xff;
		value.x = ((float)((val & 0x00ff0000U) >> 16)) / (float)0xff;
		value.y = ((float)((val & 0x0000ff00U) >> 8)) / (float)0xff;
		value.z = ((float)((val & 0x000000ffU) >> 0)) / (float)0xff;
	}
}

template <>
void GetAttribute<bool>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ bool& value)
{
	node->QueryBoolAttribute(attr, &value);
}

template <>
void GetAttribute<float>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ float& value)
{
	node->QueryFloatAttribute(attr, &value);
}

template <>
void GetAttribute<int>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ int& value)
{
	node->QueryIntAttribute(attr, &value);
}

template <>
void GetAttribute<unsigned>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ unsigned& value)
{
	node->QueryUnsignedAttribute(attr, &value);
}

#if !defined(_M_IX86)
template <>
void GetAttribute<size_t>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ size_t& value)
{
	unsigned ui;
	node->QueryUnsignedAttribute(attr, &ui);
	value = ui;
}
#endif

template <>
void GetAttribute<string>(_In_ XMLElement* node, _In_  const char* attr, _Inout_ string& value)
{
	auto attrval = node->Attribute(attr);
	if (attrval != nullptr)
	{
		value = attrval;
	}
}
void ParseNameText(tinyxml2::XMLElement * setting, const char* name, float& val, float defval)
{
	val = defval;
	auto node = setting->FirstChildElement(name);
	if (node == nullptr) return;
	node->QueryFloatText(&val);
}

void ParseNameText(tinyxml2::XMLElement * setting, const char* name, double& val, double defval)
{
	val = defval;
	auto node = setting->FirstChildElement(name);
	if (node == nullptr) return;
	node->QueryDoubleText(&val);
}

void ParseNameText(tinyxml2::XMLElement * setting, const char* name, int& val, int defval)
{
	val = defval;
	auto node = setting->FirstChildElement(name);
	if (node == nullptr) return;
	node->QueryIntText(&val);
}

void ParseNameText(tinyxml2::XMLElement * setting, const char* name, bool& val, bool defval)
{
	val = defval;
	int intval = val;
	auto node = setting->FirstChildElement(name);
	if (node == nullptr) return;
	auto error = node->QueryBoolText(&val);
	if (error != tinyxml2::XMLError::XML_SUCCESS)
	{
		error = setting->FirstChildElement(name)->QueryIntText(&intval);
		val = (intval != 0);
	}
}

#pragma endregion

std::unique_ptr<Scene> Scene::LoadSceneFromXML(const string& xml_file)
{
	uptr<Scene> pScene = make_unique<Scene>();
	pScene->LoadFromFile(xml_file);
	return pScene;
}

void Scene::LoadFromFile(const string & xml_file)
{
	using namespace DirectX;
	m_sourceDoc = make_unique<tinyxml2::XMLDocument>();
	auto& sceneDoc = *m_sourceDoc;
	auto error = sceneDoc.LoadFile(xml_file.c_str());
	
	path assetDir(xml_file);
	assetDir = assetDir.remove_filename();
	m_assets->SetAssetDirectory(assetDir);

	assert(error == XMLError::XML_SUCCESS);

	auto nScene = sceneDoc.FirstChildElement("scene");
	auto nAssets = nScene->FirstChildElement("scene.assets");

	m_settings = nScene->FirstChildElement("scene.settings");
	ParseSceneSettings(nScene);
	m_assets->ParseArchive(nAssets);

	auto nHud = nScene->FirstChildElement("scene.hud");

	auto nContent = nScene->FirstChildElement("scene.content");
	nContent = nContent->FirstChildElement();
	SceneObjectParser::ParseSceneObject(this, nullptr, nContent);

	//UpdateRenderViewCache();

	is_loaded = true;
}

void ParseSceneSettings(tinyxml2::XMLElement * nScene)
{
	auto nSettings = nScene->FirstChildElement("scene.settings");

	unsigned clipframe = 90;
	auto child = nSettings->FirstChildElement("ClipRasterizeFrame");
	if (child)
		child->QueryUnsignedText(&clipframe);
	CLIP_FRAME_COUNT = clipframe;

#define SETTING_REGISTERATION(type, name, val) ParseNameText(nSettings,#name,g_##name,val);
#include "SettingsRegisteration.h"
#undef SETTING_REGISTERATION
}


SceneObject* SceneObjectParser::ParseSceneObject(Scene* scene, SceneObject* parent, const ParamArchive* archive)
{
	using namespace DirectX;

	SceneObject* pSceneObject = nullptr;

	// parent property
	auto tag = GetArchiveName(archive);

	if (*std::find(tag, tag + strlen(tag), '.'))
		return nullptr; // This archive is a parent's property

	auto& creators = SceneObjectParser::Creators();
	auto itr = creators.find(tag);
	if (itr != creators.end())
	{
		pSceneObject = itr->second(scene, archive);
	}
	else
	{
		pSceneObject = creators["scene_object"](scene, archive);
	}

	if (parent == nullptr) // this is scene root
		scene->SetContent(pSceneObject);
	else
	{
		parent->AddChild(pSceneObject);
		std::lock_guard<mutex> guard(scene->ContentMutex());
		scene->SignalCameraCache();
	}

	archive = GetFirstChildArchive(archive);
	while (archive)
	{
		auto pChild = ParseSceneObject(scene, pSceneObject, archive);
		archive = GetNextSiblingArchive(archive);
	}
	return pSceneObject;
}
