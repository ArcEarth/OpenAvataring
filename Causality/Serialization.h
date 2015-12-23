#pragma once
#include "String.h"
#include "Math3D.h"

namespace tinyxml2
{
	class XMLElement;
	class XMLDocument;
}


namespace Causality
{
	using ParamArchive = tinyxml2::XMLElement;
	using ParamDocument = tinyxml2::XMLDocument;

	namespace Serialization
	{
		bool GetString(const ParamArchive* archive, const char* name, const char*(&param));
		bool GetString(const ParamArchive* archive, const char* name, string& param);
		bool GetBool(const ParamArchive* archive, const char* name, bool& param);
		bool GetFloat(const ParamArchive* archive, const char* name, float& param);
		bool GetDouble(const ParamArchive* archive, const char* name, double& param);
		bool GetInt(const ParamArchive* archive, const char* name, int& param);
		bool GetUint(const ParamArchive* archive, const char* name, unsigned& param);
		bool GetVector2(const ParamArchive* archive, const char* name, Vector2& param);
		bool GetVector3(const ParamArchive* archive, const char* name, Vector3& param);
		bool GetVector4(const ParamArchive* archive, const char* name, Vector4& param);
		bool GetColor(const ParamArchive* archive, const char* name, Color& param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, const char*(&param))
	{
		return Serialization::GetString(archive, name, param);
	}
	inline bool GetParam(const ParamArchive* archive, const char* name, string& param)
	{
		return Serialization::GetString(archive, name, param);
	}
	inline bool GetParam(const ParamArchive* archive, const char* name, bool& param)
	{
		return Serialization::GetBool(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, float& param)
	{
		return Serialization::GetFloat(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, double& param)
	{
		return Serialization::GetDouble(archive, name, param);
	}
	inline bool GetParam(const ParamArchive* archive, const char* name, int& param)
	{
		return Serialization::GetInt(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, unsigned& param)
	{
		return Serialization::GetUint(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, Vector2& param)
	{
		return Serialization::GetVector2(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, Vector3& param)
	{
		return Serialization::GetVector3(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, Vector4& param)
	{
		return Serialization::GetVector4(archive, name, param);
	}

	inline bool GetParam(const ParamArchive* archive, const char* name, Color& param)
	{
		return Serialization::GetColor(archive, name, param);
	}

	const char* GetArchiveName(const ParamArchive* archive);
	const ParamArchive* GetParentArchive(const ParamArchive* archive);
	const ParamArchive* GetFirstChildArchive(const ParamArchive* archive, const char* name = 0);
	const ParamArchive* GetPrevSiblingArchive(const ParamArchive* archive, const char* name = 0);
	const ParamArchive* GetNextSiblingArchive(const ParamArchive* archive, const char* name = 0);
}