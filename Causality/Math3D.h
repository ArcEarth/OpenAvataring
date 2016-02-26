#pragma once
#include <DirectXMathExtend.h>
#include <DirectXMathSimpleVectors.h>
#include <DirectXMathTransforms.h>
#include <HierarchicalTransform.h>

namespace Causality
{
	// using DirectX math as primary Math Library
	namespace Math = DirectX;

	using DirectX::Vector2;
	using DirectX::Vector3;
	using DirectX::Vector4;
	using DirectX::Quaternion;
	using DirectX::Plane;
	using DirectX::Ray;
	using DirectX::Color;
	using DirectX::Matrix4x4;
	using DirectX::DualQuaternion;

	using DirectX::AlignedNew;
	using DirectX::AlignedAllocator;
	//namespace Colors = DirectX::Colors;

	using DirectX::RigidTransform;
	using DirectX::IsometricTransform;
	using DirectX::LinearTransform;
	using DirectX::HierachicalTransform;

	using DirectX::BoundingBox;
	using DirectX::BoundingOrientedBox;
	using DirectX::BoundingFrustum;
	using DirectX::BoundingSphere;
	using DirectX::BoundingGeometry;

	// Fast SIMD Vectors
	using DirectX::XMVECTOR;
	using DirectX::FXMVECTOR;
	using DirectX::GXMVECTOR;
	using DirectX::CXMVECTOR;

	using DirectX::XMMATRIX;
	using DirectX::FXMMATRIX;
	using DirectX::CXMMATRIX;

	using DirectX::XMDUALVECTOR;
	using DirectX::FXMDUALVECTOR;
	using DirectX::GXMDUALVECTOR;
	using DirectX::CXMDUALVECTOR;

	using DirectX::XMVECTORF32;

	using DirectX::operator +;
	using DirectX::operator -;
	using DirectX::operator /;
	using DirectX::operator *;
	//using DirectX::operator *=;
	//using DirectX::operator -=;
	//using DirectX::operator +=;
	//using DirectX::operator /=;
}