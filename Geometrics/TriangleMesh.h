#pragma once

#include <DirectXMathExtend.h>
#include <vector>
#include <gsl.h>
#include <VertexTraits.h>
#include <unordered_map>

namespace Geometrics
{
	template <typename IndexType>
	struct Triangle
	{
		static constexpr size_t VertexCount = 3;
		IndexType v[VertexCount];
		Triangle() {}
		Triangle(const IndexType &v0, const IndexType &v1, const IndexType &v2)
		{
			v[0] = v0; v[1] = v1; v[2] = v2;
		}
		Triangle(const IndexType *v_)
		{
			v[0] = v_[0]; v[1] = v_[1]; v[2] = v_[2];
		}
		template <class S> explicit Triangle(const S &x)
		{
			v[0] = x[0];  v[1] = x[1];  v[2] = x[2];
		}
		IndexType &operator[] (int i) { return v[i]; }
		const IndexType &operator[] (int i) const { return v[i]; }
		operator const IndexType * () const { return &(v[0]); }
		operator IndexType * () { return &(v[0]); }

		void rewind() {
			std::swap(v[0], v[2]);
		}

		int indexof(IndexType v_) const
		{
			return (v[0] == v_) ? 0 :
				(v[1] == v_) ? 1 :
				(v[2] == v_) ? 2 : -1;
		}
	};

	template <typename _VertexType, typename _IndexType = uint16_t, typename _FaceType = Triangle<_IndexType>>
	class PolygonSoup
	{
	public:
		typedef _VertexType VertexType;
		typedef _IndexType	IndexType;
		typedef _FaceType	FaceType;

		PolygonSoup() {}

		bool empty() const {
			return vertices.empty() || indices.empty();
		}

		void clear()
		{
			vertices.clear();
			indices.clear();
		}

		inline const FaceType& facet(int idx) const
		{
			return reinterpret_cast<const FaceType&>(indices[idx * FaceType::VertexCount]);
		}

		inline FaceType& facet(int idx)
		{
			return reinterpret_cast<FaceType&>(indices[idx * FaceType::VertexCount]);
		}

		auto facets() {
			return gsl::span<FaceType>(
				reinterpret_cast<FaceType*>(indices.data()),
				indices.size() / FaceType::VertexCount);
		}

		auto facets() const {
			return gsl::span<const FaceType>(
				reinterpret_cast<const FaceType*>(indices.data()),
				indices.size() / FaceType::VertexCount);
		}

		const VertexType& vertex(int facet, int vidx) const
		{
			return this->vertices[facet * VertexCount + vidx];
		}

		VertexType& vertex(int facet, int vidx)
		{
			return this->vertices[facet * VertexCount + vidx];
		}

		std::vector<VertexType> vertices;
		std::vector<IndexType>	indices;
	};

	template <class _VertexType, class _IndexType>
	int intersect(const PolygonSoup<_VertexType, _IndexType, Triangle<_IndexType>> &Mesh, DirectX::FXMVECTOR Origin, DirectX::FXMVECTOR Direction, std::vector<float>* distances = nullptr)
	{
		using namespace DirectX;
		int count = 0;
		XMVECTOR vDir = XMVector3Normalize(Direction);
		for (const auto& tri : Mesh.facets())
		{
			float distance;

			XMVECTOR v0 = XMLoadFloat3(&Mesh.vertices[tri[0]].position);
			XMVECTOR v1 = XMLoadFloat3(&Mesh.vertices[tri[1]].position);
			XMVECTOR v2 = XMLoadFloat3(&Mesh.vertices[tri[2]].position);

			bool hr = DirectX::TriangleTests::Intersects(Origin, vDir, v0, v1, v2, distance);
			if (hr) {
				++count;
				if (distances) {
					distances->push_back(distances);
				}
			}
		}
		return count;
	}


	struct MeshRayIntersectionInfo
	{
		using Vector3 = DirectX::Vector3;

		Vector3 position;
		float	distance;
		Vector3 barycentric;
		int		facet; // facet id

		inline bool operator < (const MeshRayIntersectionInfo& rhs)
		{
			return this->distance < rhs.distance;
		}
	};

	/// <summary>
	/// Basic triangle mesh, each index represent an edge, which is the edge oppsite to the vertex in it's owner triangle
	/// </summary>
	template <typename _VertexType, typename _IndexType = uint16_t>
	class TriangleMesh : public PolygonSoup<_VertexType, _IndexType, Triangle<_IndexType>>
	{
	public:
		static const size_t VertexCount = FaceType::VertexCount;

	public:
		// edge's reverse edge
		// stores the adjacent edges of a edge in a triangle
		std::vector<IndexType> revedges;

		// vertex's first adjacant edge
		// std::vector<IndexType> vedges;

		union EdgeType
		{
			_IndexType v[2];
			struct {
				_IndexType v0, v1;
			};
		};

		inline EdgeType edge(int facet, int edge) const
		{
			EdgeType e;
			auto& tri = this->facet(facet);
			switch (edge)
			{
			case 0 :
				e.v0 = tri[1];
				e.v1 = tri[2];
				break;
			case 1 :
				e.v0 = tri[2];
				e.v1 = tri[0];
				break;
			case 2 :
				e.v0 = tri[0];
				e.v1 = tri[1];
				break;
			default:
				e.v0 = -1;
				e.v1 = -1;
				break;
			}
			return e;
		}

		inline EdgeType edge(int eid) const
		{
			return edge(eid / VertexCount, eid % VertexCount);
		}

		inline int adjacentFacet(int facet, int edge) const
		{
			return revedges[facet * VertexCount + edge] / VertexCount;
		}

		inline int adjacentFacet(int eid) const
		{
			return revedges[eid] / VertexCount;
		}

		// build the adjacent map so we can access all the 1 rings
		void build()
		{
			// intialize all adjacant edges to -1
			revedges.resize(this->indices.size(), -1);
			_IndexType vsize = this->vertices.size();
			int esize = this->indices.size();

			// make sure int for hash is enough
			assert(vsize*vsize < std::numeric_limits<int>::max());
			std::unordered_map<int, _IndexType> edges(esize * 2);
			// max items inside this table should be less than (esize / 2)

			_IndexType fid = 0;
			for (const Triangle<_IndexType>& tri : this->facets())
			{
				for (_IndexType i = 0; i < VertexCount; i++)
				{
					_IndexType eid = i + fid * 3;
					auto e = edge(fid, i);

					int ehash = e.v0 * vsize + e.v1;
					int revehash = e.v0 + e.v1 * vsize;

					auto revItr = edges.find(revehash);
					if (revItr == edges.end())
					{
						edges[ehash] = fid + 0;
					}
					else // find reversed edge, remove from edges map
					{
						auto revEid = revItr->second;
						revedges[eid] = revEid;
						revedges[revEid] = eid;
						edges.erase(revItr);
					}
				}

				++fid;
			}
		}

		int intersect(DirectX::FXMVECTOR Origin, DirectX::FXMVECTOR Direction, std::vector<MeshRayIntersectionInfo>* output) const
		{
			//assert(revedges.size() == indices.size());
			using namespace DirectX;
			size_t count = 0;
			XMVECTOR vDir = XMVector3Normalize(Direction);
			auto fid = 0;
			for (const auto& tri : this->facets())
			{
				float distance;

				XMVECTOR v0 = XMLoadFloat3(&this->vertices[tri[0]].position);
				XMVECTOR v1 = XMLoadFloat3(&this->vertices[tri[1]].position);
				XMVECTOR v2 = XMLoadFloat3(&this->vertices[tri[2]].position);

				bool hr = DirectX::TriangleTests::Intersects(Origin, vDir, v0, v1, v2, distance);
				if (hr) {
					++count;
					if (output) {
						output->emplace_back();
						auto& info = output->back();
						XMVECTOR pos = distance * vDir + Origin;
						XMVECTOR bc = DirectX::TriangleTests::BarycentricCoordinate(pos, v0, v1, v2);
						info.facet = fid;
						info.position = pos;
						info.distance = distance;
						info.barycentric = bc;
					}
				}
				++fid;
			}

			if (output)
				std::sort(output->begin(), output->end());
			return count;
		}
	};

	namespace Internal
	{
#ifndef min
		template <typename T>
		inline T clamp(T value, T minV, T maxV)
		{
			return std::max(std::min(value, maxV), minV);
		}
#else
		template <typename T>
		inline T clamp(T value, T minV, T maxV)
		{
			return max(min(value, maxV), minV);
		}
#endif
	}

	/// <summary>
	/// Compute the closest projection point from P0 to triangle(V0,V1,V2).
	/// </summary>
	/// <param name="P0">The p0.</param>
	/// <param name="V0">The v0.</param>
	/// <param name="V1">The v1.</param>
	/// <param name="V2">The v2.</param>
	/// <returns></returns>
	inline DirectX::XMVECTOR Projection(DirectX::FXMVECTOR P0, DirectX::FXMVECTOR V0, DirectX::FXMVECTOR V1, DirectX::GXMVECTOR V2)
	{
		using namespace DirectX;
		using namespace Geometrics::Internal;
		XMVECTOR edge0 = V1 - V0;
		XMVECTOR edge1 = V2 - V0;
		XMVECTOR p0 = V0 - P0;

		//XMVectorBaryCentric();

		float a = XMVectorGetX(XMVector3LengthSq(edge0));
		float b = XMVectorGetX(XMVector3Dot(edge0, edge1));
		float c = XMVectorGetX(XMVector3LengthSq(edge1));
		float d = XMVectorGetX(XMVector3Dot(edge0, p0));
		float e = XMVectorGetX(XMVector3Dot(edge1, p0));

		float det = a*c - b*b;
		float s = b*e - c*d;
		float t = b*d - a*e;

		if (s + t < det)
		{
			if (s < 0.f)
			{
				if (t < 0.f)
				{
					if (d < 0.f)
					{
						s = clamp(-d / a, 0.f, 1.f);
						t = 0.f;
					}
					else
					{
						s = 0.f;
						t = clamp(-e / c, 0.f, 1.f);
					}
				}
				else
				{
					s = 0.f;
					t = clamp(-e / c, 0.f, 1.f);
				}
			}
			else if (t < 0.f)
			{
				s = clamp(-d / a, 0.f, 1.f);
				t = 0.f;
			}
			else
			{
				float invDet = 1.f / det;
				s *= invDet;
				t *= invDet;
			}
		}
		else
		{
			if (s < 0.f)
			{
				float tmp0 = b + d;
				float tmp1 = c + e;
				if (tmp1 > tmp0)
				{
					float numer = tmp1 - tmp0;
					float denom = a - 2 * b + c;
					s = clamp(numer / denom, 0.f, 1.f);
					t = 1 - s;
				}
				else
				{
					t = clamp(-e / c, 0.f, 1.f);
					s = 0.f;
				}
			}
			else if (t < 0.f)
			{
				if (a + d > b + e)
				{
					float numer = c + e - b - d;
					float denom = a - 2 * b + c;
					s = clamp(numer / denom, 0.f, 1.f);
					t = 1 - s;
				}
				else
				{
					s = clamp(-e / c, 0.f, 1.f);
					t = 0.f;
				}
			}
			else
			{
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1.f - s;
			}
		}

		return V0 + s * edge0 + t * edge1;
	}

	inline float Distance(DirectX::FXMVECTOR P0, DirectX::FXMVECTOR V0, DirectX::FXMVECTOR V1, DirectX::GXMVECTOR V2)
	{
		using namespace DirectX;
		XMVECTOR vProj = Projection(P0, V0, V1, V2);
		vProj -= P0;
		return XMVectorGetX(XMVector3Length(vProj));
	}

	template <typename _VertexType, typename _IndexType>
	inline float Distance(const TriangleMesh<_VertexType, _IndexType> &Mesh, DirectX::FXMVECTOR Point)
	{
		using namespace DirectX;
		float minDis = numeric_limits<float>::max();
		for (const auto& tri : Mesh.facets())
		{
			float dis;

			XMVECTOR v0 = XMLoadFloat3(&Mesh.vertices[tri[0]].position);
			XMVECTOR v1 = XMLoadFloat3(&Mesh.vertices[tri[1]].position);
			XMVECTOR v2 = XMLoadFloat3(&Mesh.vertices[tri[2]].position);

			dis = Distance(Point, v0, v1, v2);
			if (dis < minDis) minDis = dis;
			//minDis = std::min(dis,minDis);
		}

		//XMVECTOR vDis = XMVectorReplicate(numeric_limits<float>::max());
		//for (const auto& vertex : Mesh.vertices)
		//{
		//	XMVECTOR vPos = XMLoadFloat3(&vertex.position);
		//	vPos -= Point;
		//	XMVECTOR dis = XMVector3Length(vPos);
		//	vDis = XMVectorMin(vDis,dis);
		//}

		//float minDisV = XMVectorGetX(vDis);
		//assert(minDisV >= minDis);
		return minDis;
		//return XMVectorGetX(minDis);
	}


	template <typename _VertexType, typename _IndexType>
	bool Inside(const TriangleMesh<_VertexType, _IndexType> &Mesh, DirectX::FXMVECTOR Point)
	{
		XMFLOAT3A Direction;
		Direction.x = (float)std::rand() / (RAND_MAX + 1);
		Direction.y = (float)std::rand() / (RAND_MAX + 1);
		Direction.z = (float)std::rand() / (RAND_MAX + 1);
		XMVECTOR vDir = XMLoadFloat3A(&Direction);
		auto count = RayIntersection(Mesh, Point, vDir, nullptr);
		return count & 1; //count % 2
	}



}