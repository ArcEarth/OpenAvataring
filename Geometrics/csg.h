#pragma once

// Original CSG.JS library by Evan Wallace (http://madebyevan.com), under the MIT license.
// GitHub: https://github.com/evanw/csg.js/
// 
// C++ port by Tomasz Dabrowski (http://28byteslater.com), under the MIT license.
// GitHub: https://github.com/dabroz/csgjs-cpp/
// 
// Constructive Solid Geometry (CSG) is a modeling technique that uses Boolean
// operations like union and intersection to combine 3D solids. This library
// implements CSG operations on meshes elegantly and concisely using BSP trees,
// and is meant to serve as an easily understandable implementation of the
// algorithm. All edge cases involving overlapping coplanar polygons in both
// solids are correctly handled.
//
// To use this as a header file, define CSGJS_HEADER_ONLY before including this file.
//

#include <vector>
#include <algorithm>
#include <cmath>
#include <DirectXMathExtend.h>
#include <cstdint>
#include "TriangleMesh.h"

namespace Geometrics
{
	using DirectX::Vector3;
	using DirectX::Color;
	using DirectX::Vector2;

	struct Vertex
	{
		Vector3 position;
		Vector3 normal;
		Vector2 uv;
		//Color	color = Color(1.0, 0.0, 0.0);
	};

	typedef uint16_t Index;

	using MeshType = TriangleMesh<Vertex, Index>;

	// public interface - not super efficient, if you use multiple CSG operations you should
	// use BSP trees and convert them into model only once. Another optimization trick is
	// replacing MeshType with your own class.
	MeshType csg_union(const MeshType & a, const MeshType & b);
	MeshType csg_intersection(const MeshType & a, const MeshType & b);
	MeshType csg_difference(const MeshType & a, const MeshType & b);
}