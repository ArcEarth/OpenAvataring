#pragma once
#include "BCL.h"
#include <filesystem>
#include "RenderSystemDecl.h"
//#include <boost\range\adaptor\map.hpp>

//#include <ppltasks.h>
#include <typeindex>
#include <unordered_map>
//#include <boost\any.hpp>
#include "Serialization.h"


//#include "Textures.h"
//#include "Models.h"  
//#include "Armature.h"
//#include "CharacterBehavier.h"
//#include <Effects.h>

namespace Causality
{
	class AssetDictionary;
	//namespace adaptors = boost::adaptors;
	//using concurrency::task;

	class ArmatureFrameAnimation;
	class BehavierSpace;
	class StaticArmature;

	// Controls the asset name resolve and loading
	class AssetDictionary
	{
	public:
		using path = std::tr2::sys::path;
		//using any = boost::any;

		using mesh_type = IModelNode;
		using texture_type = Texture;
		using audio_clip_type = int;
		using animation_clip_type = ArmatureFrameAnimation;
		using behavier_type = BehavierSpace;
		using armature_type = StaticArmature;
		using effect_type = IEffect;
		using material_type = IMaterial;
		//using any_type = any;

		AssetDictionary();
		~AssetDictionary();

		//Synchronize loading methods
		void				 ParseArchive(const ParamArchive* store);

		audio_clip_type*	 ParseAudio(const ParamArchive* store);
		mesh_type*			 ParseMesh(const ParamArchive* store);
		sptr<material_type>	 ParseMaterial(const ParamArchive* store);
		texture_type*		 ParseTexture(const ParamArchive* store);
		armature_type*		 ParseArmature(const ParamArchive* store);
		animation_clip_type* ParseAnimation(const ParamArchive* store);
		behavier_type*		 ParseBehavier(const ParamArchive* store);

		mesh_type*		     LoadObjMesh(const string & key, const string& fileName, bool flipNormal = false, const ParamArchive* aditionalOptions = nullptr);
		mesh_type*		     LoadFbxMesh(const string & key, const string& fileName, const std::shared_ptr<material_type> &pMaterial = nullptr, const ParamArchive* aditionalOptions = nullptr);
		mesh_type*		     LoadFbxMesh(const string & key, const string& fileName, bool importMaterial, const ParamArchive* aditionalOptions = nullptr);
		texture_type*	     LoadTexture(const string & key, const string& fileName);
		armature_type*	     LoadArmature(const string & key, const string& fileName, const ParamArchive* aditionalOptions = nullptr);
		animation_clip_type* LoadAnimation(const string& key, const string& fileName, const ParamArchive* aditionalOptions = nullptr);
		behavier_type*		 LoadBehavierFbx(const string & key, const string & fileName, const ParamArchive* aditionalOptions = nullptr);
		behavier_type*		 LoadBehavierFbxs(const string & key, const string& armature, list<std::pair<string, string>>& animations, const ParamArchive* aditionalOptions = nullptr);

		// Async loading methods
/*		task<mesh_type*>&			LoadMeshAsync(const string & key, const string& fileName);
		task<texture_type*>&		LoadTextureAsync(const string & key, const string& fileName);
		task<audio_clip_type*>&		LoadAudioAsync(const string & key, const string& fileName)*/;

		bool IsAssetLoaded(const string& key) const;

		//template<typename TAsset>
		//TAsset*						GetAsset(const string& key)
		//{
		//	return any_cast<TAsset*>(assets[key]);
		//}

		mesh_type*					GetMesh(const string& key)
		{
			auto itr = meshes.find(key);
			if (itr != meshes.end())
				return itr->second;
			return nullptr;
		}

		texture_type*				GetTexture(const string& key)
		{
			auto itr = textures.find(key);
			if (itr != textures.end())
				return itr->second;
			return itr->second;
		}

		animation_clip_type*		GetAnimation(const string& key)
		{
			return animations[key];
		}

		behavier_type*				GetBehavier(const string& key)
		{
			return behaviers[key];
		}

		audio_clip_type*			GetAudio(const string& key);

		effect_type*				GetEffect(const string& key)
		{
			return effects[key];
		}
		armature_type*				GetArmature(const string& key)
		{
			return armatures[key];
		}

		sptr<material_type>			GetMaterial(const string& key);

		template<typename VertexType>
		const cptr<ID3D11InputLayout>& GetInputLayout(IEffect* pEffct = nullptr);

		IEffectFactory&		EffctFactory() { return *effect_factory; }

		//template<typename TAsset>
		//TAsset&						AddAsset(const string&key, TAsset* pAsset)
		//{
		//	assets[key] = pAsset;
		//	return *pAsset;
		//}

		material_type*				AddMaterial(const string& key, const sptr<material_type>& pMaterial)
		{
			materials[key] = pMaterial;
			return pMaterial.get();
		}

		bool AddEffect(const string& key, effect_type* pEffect)
		{
			auto itr = effects.find(key);
			if (itr != effects.end())
			{
				return false;
			}
			effects[key] = pEffect;
			return true;
		}

		//template<typename TAsset>
		//TAsset&						LoadAsset(const string& fileName, function<TAsset(const string& fileName, std::istream& file)> deserializer)
		//{
		//}

		//template<typename TAsset>
		//task<TAsset&>&				LoadAssetAsync(const string& fileName, function<TAsset(const string& fileName, std::istream& file)> deserializer)
		//{
		//}

		IRenderDevice* GetRenderDevice() { return m_device.Get(); }
		const IRenderDevice* GetRenderDevice() const { return m_device.Get(); }
		void SetRenderDevice(IRenderDevice* device);
		void SetParentDictionary(AssetDictionary* dict);
		void SetTextureDirectory(const path& dir);
		const path& GetTextureDirectory() const { return texture_directory; }
		void SetMeshDirectory(const path& dir);
		void SetAssetDirectory(const path& dir);

	private:
		cptr<IRenderDevice>					m_device;

		AssetDictionary*					parent_dictionary;

		path								asset_directory;
		path								texture_directory;
		path								mesh_directory;
		path								animation_directory;

		//map<string, task<mesh_type*>>		loading_meshes;
		//map<string, task<texture_type*>>	loading_textures;

		map<string, mesh_type*>				meshes;
		map<string, texture_type*>			textures;
		map<string, animation_clip_type*>	animations;
		map<string, audio_clip_type*>		audios;
		map<string, effect_type*>			effects;
		map<string, behavier_type*>			behaviers;
		map<string, armature_type*>			armatures;
		map<string, sptr<material_type>>	materials;

		sptr<IEffect>				default_effect;
		sptr<IEffect>				default_skinned_effect;
		sptr<IEffect>				default_envirument_effect;
		sptr<PhongMaterial>			default_material;
		uptr<IEffectFactory>		effect_factory;

		map<std::type_index, cptr<ID3D11InputLayout>> layouts;

		// other assets
		//map<string, any_type>		m_assets;

	public:

		auto GetEffects()
		{
			return effects;
		}
	};

	template <typename VertexType>
	inline const cptr<ID3D11InputLayout>& AssetDictionary::GetInputLayout(IEffect * pEffct)
	{
		std::type_index type = typeid(VertexType);
		auto& pLayout = layouts[type];
		if (pLayout == nullptr)
		{
			void const* shaderByteCode;
			size_t byteCodeLength;
			pEffct->GetVertexShaderBytecode(&shaderByteCode, &byteCodeLength);
			pLayout = CreateInputLayout<VertexType>(m_device.Get(), shaderByteCode, byteCodeLength);
		}
		return pLayout;
	}

}