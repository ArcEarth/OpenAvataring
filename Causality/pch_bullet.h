#pragma once

// Prevent Windows header's macro pollution
#ifndef NOMINMAX
#define NOMINMAX
#endif

// STL and standard lib
#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <mutex>
#include <thread>
#include <random>
#include <algorithm>

// containers
#include <map>
#include <vector>
#include <list>
#include <array>
#include <deque>
#include <type_traits>

// WRL
#include <wrl.h>
#include <wrl/client.h>

// DirectX
#include <d3d11_2.h>
#include <d2d1_2.h>
#include <d2d1effects_1.h>
#include <dwrite_2.h>
#include <wincodec.h>
#include <DirectXMath.h>
#include <DirectXColors.h>
#include <DirectXCollision.h>
#include <SimpleMath.h>
#include "..\DirectX\Inc\DirectXMathExtend.h"

// Bullet physics
#ifndef BT_NO_SIMD_OPERATOR_OVERLOADS
#define BT_NO_SIMD_OPERATOR_OVERLOADS
#endif

#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>

#ifndef _InterlockedExchangePointer
#define _InterlockedExchangePointer(_Target, _Value) reinterpret_cast<void *>(static_cast<__w64 long>(_InterlockedExchange( \
    static_cast<long volatile *>(reinterpret_cast<__w64 long volatile *>(static_cast<void * volatile *>(_Target))), \
    static_cast<long>(reinterpret_cast<__w64 long>(static_cast<void *>(_Value))))))
#endif
// PPL
#include <ppl.h>
#include <ppltasks.h>