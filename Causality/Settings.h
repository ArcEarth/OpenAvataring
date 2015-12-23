#pragma once

namespace Causality
{
	const static size_t	g_PvDimension = 6U;

	enum PartAssignmentTransformMethodEnum
	{
		PAT_Default = 0,
		PAT_CCA = 1,
		PAT_OneAxisRotation = 2,
		PAT_AnisometricScale = 3,
		PAT_RST = 4,
	};

#define SETTING_REGISTERATION(type, name, val) extern type g_##name;
#include "SettingsRegisteration.h"
#undef SETTING_REGISTERATION

}