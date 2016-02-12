#pragma once
#include "BCL.h"
#include <VertexTypes.h>
#include <MeshData.h>
#include "Armature.h"
#include "CharacterBehavier.h"
#include "Serialization.h"

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

		enum class AxisSystem
		{
			MayaZUp,			/*!< UpVector = ZAxis, FrontVector = -ParityOdd, CoordSystem = RightHanded */
			MayaYUp,			/*!< UpVector = YAxis, FrontVector =  ParityOdd, CoordSystem = RightHanded */
			Max,				/*!< UpVector = ZAxis, FrontVector = -ParityOdd, CoordSystem = RightHanded */
			MotionBuilder,		/*!< UpVector = YAxis, FrontVector =  ParityOdd, CoordSystem = RightHanded */
			OpenGL,				/*!< UpVector = YAxis, FrontVector =  ParityOdd, CoordSystem = RightHanded */
			DirectX,			/*!< UpVector = YAxis, FrontVector =  ParityOdd, CoordSystem = LeftHanded */
			Lightwave,			/*!< UpVector = YAxis, FrontVector =  ParityOdd, CoordSystem = LeftHanded */
			Unknown = -1
		};

		struct FbxFileImportOptions
		{
			AxisSystem			FileAxisSystem;			// A lot of fbx did not specifies the axis system well, by default we assume it is MayaYUp system, but we can overhaul this assumption here
			unsigned			ImportMode;				// Specifies which things to import
			IsometricTransform	AdditionalTransform;	// Additonal transform applies to mesh/animation after all other things
			bool				RewindTriangles;		// Toggles to rewind the triangles
			bool				FlipNormals;			// Toggles to flip normal
		};

		explicit FbxParser(const ParamArchive* parameters);
		explicit FbxParser(const string& file, unsigned mode);

		static AxisSystem ParseAxisSystem(const char *name);

		void SetBehavierProfile(BehavierSpace* pBehav);

		bool Load(const string& file, unsigned mode);
		// Import Armature & Animations
		bool ImportBehavier(const string& file);
		// Import Armature only
		bool ImportArmature(const string& file);
		// Import Meshes only
		bool ImportMesh(const string& file, bool rewind = false, bool flipNormal = false);
		// Import Animations only
		bool ImportAnimation(const string& file,const string& name);

		void SetAdditionalTransform(const IsometricTransform& transform);
		void SetFileAxisSystemOverhaul(AxisSystem axis = AxisSystem::Unknown);

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