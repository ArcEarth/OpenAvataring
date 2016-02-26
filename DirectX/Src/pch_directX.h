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

// DirectX Math and extension
#include <DirectXMath.h>
#include <DirectXColors.h>
#include <DirectXCollision.h>
#include <SimpleMath.h>

#include "DirectXMathExtend.h"
#include "DirectXMathIntrinsics.h"
#include "DirectXMathSimpleVectors.h"
#include "DirectXMathTransforms.h"

//DirectX Toolkit
#include <Effects.h>
#include <CommonStates.h>
#include <VertexTypes.h>
#include <PrimitiveBatch.h>
#include <GeometricPrimitive.h>

// PPL
#include <ppltasks.h>