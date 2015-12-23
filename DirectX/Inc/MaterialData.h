#pragma once
#include "DirectXMathExtend.h"
#include <string>
//#include <ios>

namespace DirectX
{
	namespace Scene
	{
		struct MaterialData
		{
			std::string Name;

			// Overall opticity
			float		Alpha;
			// Surface micro-geometry enchancement
			std::string NormalMapName;
			std::string DisplaceMapName;

			MaterialData()
			{
				Alpha = 1.0f;
			}

		};

		struct LambertMaterialData : public MaterialData
		{
			Color		AmbientColor;
			Color		DiffuseColor;
			Color		EmissiveColor;

			std::string AmbientMapName;
			std::string DiffuseMapName;
			std::string EmissiveMapName;

			LambertMaterialData()
			{
				AmbientColor = Colors::White.v;
				DiffuseColor = Colors::White.v;
				EmissiveColor = Colors::Black.v;
			}
		};

		struct PhongMaterialData : public LambertMaterialData
		{
			Color		SpecularColor;
			Color		RelfectionColor;

			PhongMaterialData()
			{
				SpecularColor = Colors::Black.v;
				RelfectionColor = Colors::Black.v;
				UseAlphaDiscard = false;
			}

			std::string SpecularMapName;
			std::string RelfectionMapName;
			bool		UseAlphaDiscard;
		};

	}
}