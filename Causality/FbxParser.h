#pragma once
#include "BCL.h"
#include <VertexTypes.h>
#include <MeshData.h>
#include "Armature.h"
#include "CharacterBehavier.h"

namespace Causality
{
	using DirectX::Scene::SkinMeshData;
	using DirectX::Scene::StaticMeshData;

	class FbxParser
	{
	public:
		enum class Mode : unsigned
		{
			None = 0U,
			ImportMeshs = 1U,
			ImportAnimations = 2U,
			CreateBehavierAndArmature = 4U,
			ImportArmature = 8U
		};

		explicit FbxParser(const string& file, unsigned mode);

		void SetBehavierProfile(BehavierSpace* pBehav);

		bool Load(const string& file, unsigned mode);
		// Import Armature & Animations
		bool ImportBehavier(const string& file);
		// Import Armature only
		bool ImportArmature(const string& file);
		// Import Meshes only
		bool ImportMesh(const string& file, bool rewind = false);
		// Import Animations only
		bool ImportAnimation(const string& file,const string& name);

		const std::list<StaticMeshData>& GetStaticMeshs();

		const std::list<SkinMeshData>&	GetMeshs();
		StaticArmature*					GetArmature();
		BehavierSpace*					GetBehavier();

		~FbxParser();
		FbxParser();
	private:
		struct Impl;
		unique_ptr<Impl> m_pImpl;
	};
}