#include "Settings.h"

namespace Causality
{
#define SETTING_REGISTERATION(type, name, val) type g_##name = val;
#include "SettingsRegisteration.h"
#undef SETTING_REGISTERATION
}