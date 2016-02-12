#include "pch_bcl.h"
#include "Serialization.h"
#include <tinyxml2.h>
#include <sstream>

using namespace Causality;
using namespace Causality::Serialization;
using namespace tinyxml2;
using namespace std;

namespace Causality
{
	namespace Serialization
	{
		bool GetString(const ParamArchive * store, const char * name, const char *(&param))
		{
			if (!store) return false;
			auto str = store->Attribute(name);
			if (str != nullptr)
			{
				param = str;
				return true;
			}
			else
			{
				store = store->FirstChildElement(name);
				if (store != nullptr)
				{
					param = store->GetText();
					return true;
				}
			}
			return false;
		}

		bool GetString(const ParamArchive * store, const char * name, string & param)
		{
			if (!store) return false;
			auto str = store->Attribute(name);
			if (str != nullptr)
			{
				param = str;
				return true;
			}
			else
			{
				store = store->FirstChildElement(name);
				if (store != nullptr)
				{
					param = store->GetText();
					return true;
				}
			}
			return false;
		}

#define GetTypedParam(Type) \
			if (!store) return false; \
			if (store->Query##Type##Attribute(name, &param)) \
			{ \
				store = store->FirstChildElement(name); \
				return store != nullptr && !store->Query##Type##Text(&param); \
			} \
			return false;

		bool GetBool(const ParamArchive * store, const char * name, bool & param)
		{
			GetTypedParam(Bool);
		}

		bool GetFloat(const ParamArchive * store, const char * name, float & param)
		{
			GetTypedParam(Float);
		}

		bool GetDouble(const ParamArchive * store, const char * name, double & param)
		{
			GetTypedParam(Double);
		}

		bool GetInt(const ParamArchive * store, const char * name, int & param)
		{
			GetTypedParam(Int);
		}

		bool GetUint(const ParamArchive * store, const char * name, unsigned & param)
		{
			GetTypedParam(Unsigned);
		}

		bool ParseVector2(const char* attrval, Vector2 & param)
		{
			if (attrval == nullptr) return false;
			string str(attrval);
			if (str.find_first_of(',') != string::npos)
			{
				stringstream ss(str);
				char ch;
				ss >> param.x >> ch >> param.y;
				return true;
			}
			else
			{
				param.x = param.y = (float)atof(attrval); // replicate
				return true;
			}
		}

		bool ParseVector3(const char* attrval, Vector3 & param)
		{
			if (attrval == nullptr) return false;
			string str(attrval);
			if (str.find_first_of(',') != string::npos)
			{
				stringstream ss(str);
				char ch;
				ss >> param.x >> ch >> param.y >> ch >> param.z;
				return true;
			}
			else
			{
				param.x = param.y = param.z = (float)atof(attrval); // replicate
				return true;
			}
		}

		bool ParseVector4(const char* attrval, Vector4 & param)
		{
			if (attrval == nullptr) return false;
			string str(attrval);
			if (str.find_first_of(',') != string::npos)
			{
				stringstream ss(str);
				char ch;
				ss >> param.x >> ch >> param.y >> ch >> param.z >> ch >> param.w;
				return true;
			}
			else
			{
				param.x = param.y = param.z = param.w = (float)atof(attrval); // replicate
				return true;
			}
		}

		bool ParseColor(const char* attrval, Color & param)
		{
			if (attrval == nullptr) return false;
			if (attrval != nullptr && attrval[0] == '#') // for hex code literal
			{
				char* end;
				auto val = strtoul(attrval + 1, &end, 16);
				param.w = ((float)((val & 0xff000000U) >> 24)) / (float)0xff;
				param.x = ((float)((val & 0x00ff0000U) >> 16)) / (float)0xff;
				param.y = ((float)((val & 0x0000ff00U) >> 8)) / (float)0xff;
				param.z = ((float)((val & 0x000000ffU) >> 0)) / (float)0xff;
				return true;
			}
			else if (attrval != nullptr && (std::isdigit(attrval[0]) || attrval[0] == '.')) // for Vector4 literal
			{
				return ParseVector4(attrval, reinterpret_cast<Vector4&>(param));
			}
			return false;
		}

		bool GetVector2(const ParamArchive * store, const char * name, Vector2 & param)
		{
			const char* attrval = nullptr;
			return GetString(store, name, attrval) && ParseVector2(attrval, param);
		}

		bool GetVector3(const ParamArchive * store, const char * name, Vector3 & param)
		{
			const char* attrval = nullptr;
			return GetString(store, name, attrval) && ParseVector3(attrval, param);
		}

		bool GetVector4(const ParamArchive * store, const char * name, Vector4 & param)
		{
			const char* attrval = nullptr;
			return GetString(store, name, attrval) && ParseVector4(attrval, param);
		}

		bool GetColor(const ParamArchive * store, const char * name, Color & param)
		{
			const char* attrval = nullptr;
			return GetString(store, name, attrval) && ParseColor(attrval, param);
		}
	}

	const ParamArchive * GetFirstChildArchive(const ParamArchive * store, const char * name)
	{
		return store->FirstChildElement(name);
	}

	const ParamArchive * GetPrevSiblingArchive(const ParamArchive * store, const char * name)
	{
		return store->PreviousSiblingElement(name);
	}

	const ParamArchive * GetNextSiblingArchive(const ParamArchive * store, const char * name)
	{
		return store->NextSiblingElement(name);
	}

	const ParamArchive * GetParentArchive(const ParamArchive * store)
	{
		return dynamic_cast<const XMLElement*>(store->Parent());
	}

	const char * GetArchiveName(const ParamArchive * store)
	{
		return store->Name();
	}

}

