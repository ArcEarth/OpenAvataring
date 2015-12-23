#pragma once
#ifndef DX_MATH_EXT_H
#include "DirectXMathExtend.h"
#endif

namespace DirectX
{
	template <typename GeometryType>
	inline ContainmentType BoundingGeometry::ContainsGeometry(const GeometryType & geometry) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Contains(geometry);
		case Geometry_OrientedBox:
			return OrientedBox.Contains(geometry);
		case Geometry_Frustum:
			return Frustum.Contains(geometry);
		case Geometry_Sphere:
			return Sphere.Contains(geometry);
		default:
			return ContainmentType::DISJOINT;
		}
	}
	template <typename GeometryType>
	inline bool BoundingGeometry::IntersectsGeometry(const GeometryType & geometry) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Intersects(geometry);
		case Geometry_OrientedBox:
			return OrientedBox.Intersects(geometry);
		case Geometry_Frustum:
			return Frustum.Intersects(geometry);
		case Geometry_Sphere:
			return Sphere.Intersects(geometry);
		default:
			return false;
		}
	}
	inline BoundingGeometry& BoundingGeometry::operator=(const BoundingBox& rhs)
	{
		Type = Geometry_AxisAlignedBox;
		AxisAlignedBox = rhs;
	}
	inline BoundingGeometry& BoundingGeometry::operator=(const BoundingOrientedBox& rhs)
	{
		Type = Geometry_OrientedBox;
		OrientedBox = rhs;
	}
	inline BoundingGeometry& BoundingGeometry::operator=(const BoundingFrustum& rhs)
	{
		Type = Geometry_Frustum;
		Frustum = rhs;
	}
	inline BoundingGeometry& BoundingGeometry::operator=(const BoundingSphere& rhs)
	{
		Type = Geometry_Sphere;
		Sphere = rhs;
	}

	inline BoundingGeometry::BoundingGeometry()
		: Type(Geometry_Unknown)
	{}

	inline BoundingGeometry::BoundingGeometry(const BoundingBox & aabb)
		: Type(Geometry_AxisAlignedBox), AxisAlignedBox(aabb)
	{}

	inline BoundingGeometry::BoundingGeometry(const BoundingOrientedBox & obox)
		: Type(Geometry_OrientedBox), OrientedBox(obox)
	{}

	inline BoundingGeometry::BoundingGeometry(const BoundingFrustum & frustum)
		: Type(Geometry_Frustum), Frustum(frustum)
	{}

	inline BoundingGeometry::BoundingGeometry(const BoundingSphere & sphere)
		: Type(Geometry_Sphere), Sphere(sphere)
	{}

	inline ContainmentType XM_CALLCONV BoundingGeometry::Contains(FXMVECTOR Point) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Contains(Point);
		case Geometry_OrientedBox:
			return OrientedBox.Contains(Point);
		case Geometry_Frustum:
			return Frustum.Contains(Point);
		case Geometry_Sphere:
			return Sphere.Contains(Point);
		default:
			return ContainmentType::DISJOINT;
		}
	}

	inline ContainmentType XM_CALLCONV BoundingGeometry::Contains(FXMVECTOR V0, FXMVECTOR V1, FXMVECTOR V2) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Contains(V0, V1, V2);
		case Geometry_OrientedBox:
			return OrientedBox.Contains(V0, V1, V2);
		case Geometry_Frustum:
			return Frustum.Contains(V0, V1, V2);
		case Geometry_Sphere:
			return Sphere.Contains(V0, V1, V2);
		default:
			return ContainmentType::DISJOINT;
		}
	}

	inline ContainmentType BoundingGeometry::Contains(const BoundingSphere & geometry) const
	{
		return ContainsGeometry(geometry);
	}

	inline ContainmentType BoundingGeometry::Contains(const BoundingBox & geometry) const
	{
		return ContainsGeometry(geometry);
	}

	inline ContainmentType BoundingGeometry::Contains(const BoundingOrientedBox & geometry) const
	{
		return ContainsGeometry(geometry);
	}

	inline ContainmentType BoundingGeometry::Contains(const BoundingFrustum & geometry) const
	{
		return ContainsGeometry(geometry);
	}

	inline ContainmentType BoundingGeometry::Contains(const BoundingGeometry & geometry) const
	{
		switch (geometry.Type)
		{
		case Geometry_AxisAlignedBox:
			return ContainsGeometry(geometry.AxisAlignedBox);
		case Geometry_OrientedBox:
			return ContainsGeometry(geometry.OrientedBox);
		case Geometry_Frustum:
			return ContainsGeometry(geometry.Frustum);
		case Geometry_Sphere:
			return ContainsGeometry(geometry.Sphere);
		default:
			return ContainmentType::DISJOINT;
		}
	}

	// Six-Plane containment test

	inline ContainmentType XM_CALLCONV BoundingGeometry::ContainedBy(FXMVECTOR Plane0, FXMVECTOR Plane1, FXMVECTOR Plane2, GXMVECTOR Plane3, HXMVECTOR Plane4, HXMVECTOR Plane5) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.ContainedBy(Plane0, Plane1, Plane2, Plane3, Plane4, Plane5);
		case Geometry_OrientedBox:
			return OrientedBox.ContainedBy(Plane0, Plane1, Plane2, Plane3, Plane4, Plane5);
		case Geometry_Frustum:
			return Frustum.ContainedBy(Plane0, Plane1, Plane2, Plane3, Plane4, Plane5);
		case Geometry_Sphere:
			return Sphere.ContainedBy(Plane0, Plane1, Plane2, Plane3, Plane4, Plane5);
		default:
			return ContainmentType::DISJOINT;
		}
	}

	inline bool BoundingGeometry::Intersects(const BoundingSphere & geometry) const
	{
		return IntersectsGeometry(geometry);
	}

	inline bool BoundingGeometry::Intersects(const BoundingBox & geometry) const
	{
		return IntersectsGeometry(geometry);
	}

	inline bool BoundingGeometry::Intersects(const BoundingOrientedBox & geometry) const
	{
		return IntersectsGeometry(geometry);
	}

	inline bool BoundingGeometry::Intersects(const BoundingFrustum & geometry) const
	{
		return IntersectsGeometry(geometry);
	}

	inline bool BoundingGeometry::Intersects(const BoundingGeometry & geometry) const
	{
		switch (geometry.Type)
		{
		case Geometry_AxisAlignedBox:
			return IntersectsGeometry(geometry.AxisAlignedBox);
		case Geometry_OrientedBox:
			return IntersectsGeometry(geometry.OrientedBox);
		case Geometry_Frustum:
			return IntersectsGeometry(geometry.Frustum);
		case Geometry_Sphere:
			return IntersectsGeometry(geometry.Sphere);
		default:
			return false;
		}
	}

	// Triangle test

	inline bool XM_CALLCONV BoundingGeometry::Intersects(FXMVECTOR V0, FXMVECTOR V1, FXMVECTOR V2) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Intersects(V0, V1, V2);
		case Geometry_OrientedBox:
			return OrientedBox.Intersects(V0, V1, V2);
		case Geometry_Frustum:
			return Frustum.Intersects(V0, V1, V2);
		case Geometry_Sphere:
			return Sphere.Intersects(V0, V1, V2);
		default:
			return false;
		}
	}

	// Ray test

	inline bool XM_CALLCONV BoundingGeometry::Intersects(FXMVECTOR Origin, FXMVECTOR Direction, float & Dist) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Intersects(Origin, Direction, Dist);
		case Geometry_OrientedBox:
			return OrientedBox.Intersects(Origin, Direction, Dist);
		case Geometry_Frustum:
			return Frustum.Intersects(Origin, Direction, Dist);
		case Geometry_Sphere:
			return Sphere.Intersects(Origin, Direction, Dist);
		default:
			return false;
		}
	}

	// Plane test

	inline PlaneIntersectionType XM_CALLCONV BoundingGeometry::Intersects(FXMVECTOR Plane) const
	{
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			return AxisAlignedBox.Intersects(Plane);
		case Geometry_OrientedBox:
			return OrientedBox.Intersects(Plane);
		case Geometry_Frustum:
			return Frustum.Intersects(Plane);
		case Geometry_Sphere:
			return Sphere.Intersects(Plane);
		default:
			return PlaneIntersectionType::BACK;
		}
	}
	inline void XM_CALLCONV BoundingGeometry::Transform(BoundingGeometry & Out, FXMMATRIX M) const
	{
		Out.Type = Type;
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			AxisAlignedBox.Transform(Out.AxisAlignedBox, M);
			break;
		case Geometry_OrientedBox:
			OrientedBox.Transform(Out.OrientedBox, M);
			break;
		case Geometry_Frustum:
			Frustum.Transform(Out.Frustum, M);
			break;
		case Geometry_Sphere:
			Sphere.Transform(Out.Sphere, M);
			break;
		default:
			break;
		}
	}
	inline void XM_CALLCONV BoundingGeometry::Transform(BoundingGeometry & Out, float Scale, FXMVECTOR Rotation, FXMVECTOR Translation) const
	{
		Out.Type = Type;
		switch (Type)
		{
		case Geometry_AxisAlignedBox:
			AxisAlignedBox.Transform(Out.AxisAlignedBox, Scale, Rotation, Translation);
			break;
		case Geometry_OrientedBox:
			OrientedBox.Transform(Out.OrientedBox, Scale, Rotation, Translation);
			break;
		case Geometry_Frustum:
			Frustum.Transform(Out.Frustum, Scale, Rotation, Translation);
			break;
		case Geometry_Sphere:
			Sphere.Transform(Out.Sphere, Scale, Rotation, Translation);
			break;
		default:
			break;
		}
	}

	namespace LineSegmentTest
	{
		inline float XM_CALLCONV Distance(FXMVECTOR p, FXMVECTOR s0, FXMVECTOR s1)
		{
			XMVECTOR s = s1 - s0;
			XMVECTOR v = p - s0;
			auto Ps = XMVector3Dot(v, s);
			//p-s0 is the shortest distance
			if (XMVector4LessOrEqual(Ps, XMVectorZero()))
				return XMVectorGetX(XMVector3Length(v));

			auto Ds = XMVector3LengthSq(s);
			//p-s1 is the shortest distance
			if (XMVector4Greater(Ps, Ds))
				return XMVectorGetX(XMVector3Length(p - s1));
			//find the projection point on line segment U
			return XMVectorGetX(XMVector3Length(v - s * (Ps / Ds)));
		}

		inline float XM_CALLCONV Distance(FXMVECTOR p, const XMFLOAT3 *path, size_t nPoint, size_t strideInByte)
		{
			const auto N = nPoint;
			auto ptr = reinterpret_cast<const char*>(path);
			XMVECTOR vBegin = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(path));
			XMVECTOR vEnd;
			XMVECTOR vMinDis = g_XMInfinity;

			for (size_t i = 2; i < N - 1; i++)
			{
				ptr += strideInByte;
				vEnd = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(ptr));

				XMVECTOR s = vEnd - vBegin;
				XMVECTOR v = p - vBegin;
				XMVECTOR Ps = XMVector3Dot(v, s);
				XMVECTOR vDis;

				//p-s0 is the shortest distance
				if (XMVector4LessOrEqual(Ps, XMVectorZero()))
					vDis = XMVector3Length(v);
				else {
					XMVECTOR Ds = XMVector3LengthSq(s);
					//p-s1 is the shortest distance
					if (XMVector4Greater(Ps, Ds))
						vDis = XMVector3Length(p - vEnd);
					else
						//find the projection point on line segment U
						vDis = XMVector3Length(v - s * (Ps / Ds));
				}

				vMinDis = XMVectorMin(vDis, vMinDis);

				vBegin = vEnd;
			}

			return XMVectorGetX(vMinDis);
		}

		// Takes a space point and space line segment , return the projection point on the line segment
		//  A0  |		A1		  |
		//      |s0-------------s1|
		//      |				  |		A2
		// where p in area of A0 returns s0 , area A2 returns s1 , point in A1 returns the really projection point 
		inline XMVECTOR XM_CALLCONV Projection(FXMVECTOR p, FXMVECTOR s0, FXMVECTOR s1)
		{
			XMVECTOR s = s1 - s0;
			XMVECTOR v = p - s0;
			XMVECTOR Ps = XMVector3Dot(v, s);

			//p-s0 is the shortest distance
			//if (Ps<=0.0f) 
			//	return s0;
			if (XMVector4LessOrEqual(Ps, g_XMZero.v))
				return s0;
			XMVECTOR Ds = XMVector3LengthSq(s);
			//p-s1 is the shortest distance
			//if (Ps>Ds) 	
			//	return s1;
			if (XMVector4Less(Ds, Ps))
				return s1;
			//find the projection point on line segment U
			return (s * (Ps / Ds)) + s0;
		}

		template <typename T>
		inline XMVECTOR XM_CALLCONV Projection(FXMVECTOR p, const T *path, size_t nPoint, size_t strideInByte)
		{
			if (!(strideInByte % 16) && !(reinterpret_cast<intptr_t>(path) & 0x16))
			{
				const auto N = nPoint;
				auto ptr = reinterpret_cast<const char*>(path) + strideInByte;
				XMVECTOR vBegin = XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(path));
				XMVECTOR vEnd = XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(ptr));
				XMVECTOR vMinProj = Projection(p, vBegin, vEnd);
				XMVECTOR vMinDis = XMVector3LengthSq(p - vMinProj);
				vBegin = vEnd;

				for (size_t i = 2; i < N - 1; i++)
				{
					ptr += strideInByte;
					vEnd = XMLoadFloat3A(reinterpret_cast<const XMFLOAT3A*>(ptr));
					XMVECTOR vProj = Projection(p, vBegin, vEnd);
					XMVECTOR vDis = XMVector3LengthSq(p - vProj);
					if (XMVector4LessOrEqual(vDis, vMinDis))
					{
						vMinDis = vDis;
						vMinProj = vProj;
					}
					vBegin = vEnd;
				}

				return vMinProj;
			}
			else
			{
				const auto N = nPoint;
				auto ptr = reinterpret_cast<const char*>(path) + strideInByte;
				XMVECTOR vBegin = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(path));
				XMVECTOR vEnd = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(ptr));
				XMVECTOR vMinProj = Projection(p, vBegin, vEnd);
				XMVECTOR vMinDis = XMVector3LengthSq(p - vMinProj);
				vBegin = vEnd;

				for (size_t i = 2; i < N - 1; i++)
				{
					ptr += strideInByte;
					vEnd = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(ptr));
					XMVECTOR vProj = Projection(p, vBegin, vEnd);
					XMVECTOR vDis = XMVector3LengthSq(p - vProj);
					if (XMVector4LessOrEqual(vDis, vMinDis))
					{
						vMinDis = vDis;
						vMinProj = vProj;
					}
					vBegin = vEnd;
				}

				return vMinProj;
			}
		}

	}

	namespace BoundingFrustumExtension
	{
		inline void CreateFromMatrixRH(BoundingFrustum& Out, CXMMATRIX Projection)
		{
			// Corners of the projection frustum in homogenous space.
			static XMVECTORF32 HomogenousPoints[6] =
			{
				{ 1.0f,  0.0f, -1.0f, 1.0f },   // right (at far plane)
				{ -1.0f,  0.0f, -1.0f, 1.0f },   // left
				{ 0.0f,  1.0f, -1.0f, 1.0f },   // top
				{ 0.0f, -1.0f, -1.0f, 1.0f },   // bottom

				{ 0.0f, 0.0f, 1.0f, 1.0f },    // far
				{ 0.0f, 0.0f, 0.0f, 1.0f }     // near
			};

			XMVECTOR Determinant;
			XMMATRIX matInverse = XMMatrixInverse(&Determinant, Projection);

			// Compute the frustum corners in world space.
			XMVECTOR Points[6];

			for (size_t i = 0; i < 6; ++i)
			{
				// Transform point.
				Points[i] = XMVector4Transform(HomogenousPoints[i], matInverse);
			}

			Out.Origin = XMFLOAT3(0.0f, 0.0f, 0.0f);
			Out.Orientation = XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f);

			// Compute the slopes.
			Points[0] = Points[0] * XMVectorReciprocal(XMVectorSplatZ(Points[0]));
			Points[1] = Points[1] * XMVectorReciprocal(XMVectorSplatZ(Points[1]));
			Points[2] = Points[2] * XMVectorReciprocal(XMVectorSplatZ(Points[2]));
			Points[3] = Points[3] * XMVectorReciprocal(XMVectorSplatZ(Points[3]));

			Out.RightSlope = XMVectorGetX(Points[0]);
			Out.LeftSlope = XMVectorGetX(Points[1]);
			Out.TopSlope = XMVectorGetY(Points[2]);
			Out.BottomSlope = XMVectorGetY(Points[3]);

			// Compute near and far.
			Points[4] = Points[4] * XMVectorReciprocal(XMVectorSplatW(Points[4]));
			Points[5] = Points[5] * XMVectorReciprocal(XMVectorSplatW(Points[5]));

			Out.Near = XMVectorGetZ(Points[4]);
			Out.Far = XMVectorGetZ(Points[5]);
		}
	}

}