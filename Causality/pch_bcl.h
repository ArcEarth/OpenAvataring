#ifndef _PCH_BCL_
#define _PCH_BCL_
#pragma once

#include <thread>
#include <iostream>
#include <mutex>

#ifdef NOMINMAX
#include <windows.h>
#else
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif
#include <d3d11_2.h>
#include <d2d1_2.h>
#include <d2d1effects_1.h>
#include <dwrite_2.h>

#include "bcl.h"
#include <ppltasks.h>

#endif // _PCH_BCL_