#define NOMINMAX
#include "Extrusion.h"

using namespace Geometrics;
using namespace DirectX;
using namespace std;

typedef uint16_t IndexType;

Extrusion::Extrusion()
	: m_top(nullptr), m_bottom(nullptr), m_path(nullptr), m_dirty(1)
{

}

Extrusion::Extrusion(SurfacePatch * top, SurfacePatch * bottom, Curve * axis)
	: m_top(top), m_bottom(bottom), m_path(axis), m_dirty(1)
{}

Extrusion::~Extrusion()
{

}
//
//int Extrusion::intersect(const Ray & ray, std::vector<float>* intersections)
//{
//	return m_mesh.intersect(ray.position, ray.direction, intersections);
//}

MeshType & Extrusion::mesh() { assert(!m_dirty); return m_mesh; }

const MeshType & Extrusion::mesh() const { assert(!m_dirty); return m_mesh; }

void XM_CALLCONV centralize(_Inout_ Curve & curve, FXMVECTOR center, FXMVECTOR rotation, bool rotateRespectPolarAngle)
{
	using namespace DirectX;

	auto n = curve.size();

	// polar angles in X-Z plane
	std::vector<float> angles(n);
	float minA = XM_2PI;
	int minAidx = 0;

	// zero mean and rotate with align to identity axis
	for (int i = 0; i < curve.size(); i++)
	{
		auto& ach = reinterpret_cast<XMFLOAT3A&>(curve.anchor(i));

		XMVECTOR vr = curve.position(i) - center;
		vr = XMVector3Rotate(vr, rotation);

		XMStoreFloat3A(&ach, vr);

		float ang = atan2f(ach.z, ach.x);
		angles[i] = ang;

		if (ang < minA)
		{
			minA = ang;
			minAidx = i;
		}
	}

	if (rotateRespectPolarAngle)
	{
		Curve temp(curve);
		for (int i = 0; i < n; i++)
		{
			curve.anchor(i) = temp.anchor((minAidx + i) % n);
		}

		curve.updateLength();
	}

}

RigidTransform XM_CALLCONV centralize(Curve& curve, FXMVECTOR identityAxis = DirectX::g_XMIdentityR1.v, bool rotateRespectPolarAngle = true);

/// <summary>
///	return the transform from centralized curve to original curve
/// </summary>
/// <param name="curve">curve to transform</param>
/// <param name="identityAxis">the axis that rotate the patch normal into</param>
/// <param name="rotateRespectPolarAngle">should the function zerolize the polar angle</param>
/// <returns>return the transform from centralized curve to original curve r[0] = translation, r[1] = rotation quaternion</returns>
RigidTransform XM_CALLCONV centralize(_Inout_ Curve & curve, FXMVECTOR identityAxis, bool rotateRespectPolarAngle)
{
	using namespace DirectX;

	auto n = curve.size();

	// caculate geo center
	XMVECTOR center = XMVectorZero();
	for (int i = 0; i < n; i++)
	{
		center += curve.position(i);
	}
	center /= n;

	// caculate average 'patch' normal
	XMVECTOR axis = XMVectorZero();
	for (int i = 0; i < n; i++)
	{
		XMVECTOR vr = curve.position(i) - center;
		XMVECTOR vt = curve.tangent(i);
		vt = XMVector3Cross(vr, vt);
		axis += vt;
	}
	axis = XMVector3Normalize(axis);

	XMVECTOR q = XMQuaternionRotationVectorToVector(axis, identityAxis);

	centralize(curve, center, q, rotateRespectPolarAngle);

	RigidTransform transform;
	transform.Translation = center;
	transform.Rotation = XMQuaternionInverse(q);
	return transform;
}

template <typename IndexType>
std::enable_if_t<std::is_integral<IndexType>::value> pushTriangle(vector<IndexType>& indices, IndexType a, IndexType b, IndexType c)
{
	indices.push_back(a);
	indices.push_back(b);
	indices.push_back(c);
}
// triangulate top and bottom with path into m_mesh
MeshType & Extrusion::triangulate(int axisSubdiv, int polarSubdiv)
{
	using namespace DirectX;
	XMVECTOR stdAxis = g_XMIdentityR1.v;

	bool useTop = m_top != nullptr;

	m_mesh.clear();
	assert(m_bottom != nullptr);

	auto path = *m_path;

	path.resample(axisSubdiv + 1);
	//path.smooth(0.8f, 8);

	XMVECTOR p0 = path.position(0);;
	XMVECTOR t0 = path.tangent(0);
	if (XMVector4Less(t0, g_XMEpsilon.v))
		t0 = stdAxis;

	XMVECTOR r0 = XMQuaternionRotationVectorToVector(stdAxis, t0);

	// inverse r0
	r0 = XMQuaternionConjugate(r0);

	Curve top, bottom = m_bottom->boundry();

	bottom.resample(polarSubdiv);
	centralize(bottom, p0, r0, true);

	if (useTop)
	{
		XMVECTOR p1 = path.tangent((int)(path.size() - 1));
		XMVECTOR t1 = path.tangent((int)(path.size() - 1));
		XMVECTOR r1 = XMQuaternionRotationVectorToVector(t0, t1);
		r1 = XMQuaternionMultiply(r0, r1);
		r1 = XMQuaternionConjugate(r1);

		top.resample(polarSubdiv);
		centralize(top, p1, r1, true);
	}

	// put r0 back , r1 are useless after
	r0 = XMQuaternionConjugate(r0);

	auto  length = path.length();
	auto  interval = length / axisSubdiv;
	auto& vertices = m_mesh.vertices;
	auto& indices = m_mesh.indices;

	vertices.reserve((axisSubdiv + 1) * (polarSubdiv + 1));
	indices.reserve(3 * axisSubdiv * polarSubdiv * 2);

	int idxBase = vertices.size();

	XMVECTOR r0t = XMQuaternionIdentity();
	XMVECTOR tprev = t0;
	for (int axisIdx = 0; axisIdx <= axisSubdiv; axisIdx++)
	{
		float t = axisIdx * interval;
		XMVECTOR tv = XMVectorReplicate(t);

		// axis posion
		XMVECTOR pt = path.position(axisIdx);
		// axis tanget
		XMVECTOR tt = path.tangent(axisIdx);

		// update rotation only if the tangent are not zero
		if (XMVector4Greater(tt, g_XMEpsilon.v))
		{
			XMVECTOR rtprevt = XMQuaternionRotationVectorToVector(tprev, tt);
			r0t = XMQuaternionMultiply(r0t, rtprevt);
			tprev = tt;
		}

		// axis rotation
		XMVECTOR rt = XMQuaternionMultiply(r0, r0t);

		// [0,polarSubdiv], 0 and polarSubdiv should be overlaped
		for (int i = 0; i <= polarSubdiv; ++i)
		{
			XMVECTOR v, vt;
			float polart = (float)i / (float)(polarSubdiv + 1);

			v = bottom.position(i);
			vt = bottom.tangent(i);
			if (useTop)
			{
				// Lerp between bottom curve and top curve
				v = XMVectorLerpV(v, top.position(i), tv);
				vt = XMVectorLerpV(vt, top.tangent(i), tv);
			}

			v = XMVector3Rotate(v, rt);
			v += pt;
			vt = XMVector3Rotate(vt, rt);

			Vertex vtx;
			vtx.position = v;
			vtx.normal = XMVector3Cross(tt, vt);
			vtx.uv = Vector2(polart, t);

			vertices.push_back(vtx);
		}
	}

	int cols = polarSubdiv + 1;
	for (int axisIdx = 0; axisIdx < axisSubdiv; axisIdx++)
	{
		for (int i = 0; i < polarSubdiv; ++i)
		{
			IndexType idx = axisIdx * cols + i;
			IndexType upIdx = idx + cols;

			IndexType idx_1 = axisIdx * cols + (i + 1)/* % polarSubdiv*/;
			IndexType upIdx_1 = idx_1 + cols;

			pushTriangle(indices, idx, idx_1, upIdx);
			pushTriangle(indices, idx_1, upIdx_1, upIdx);
		}
	}

	//m_mesh.build();

	m_dirty = false;
	return m_mesh;
}

//inline Curve & Geometrics::SurfacePatch::boundry() { return m_boundry; }

const Curve & SurfacePatch::boundry() const { return m_boundry; }

MeshType & SurfacePatch::mesh() { return m_mesh; }

const MeshType & SurfacePatch::mesh() const { return m_mesh; }

void SurfacePatch::clear()
{
	m_boundry.clear();
	m_uvCurve.clear();
}

bool SurfacePatch::append(XMVECTOR position, int fid)
{
	if (m_boundry.append(position))
	{
		auto& tri = m_surface->facet(fid);
		auto& vertics = m_surface->vertices;

		auto BC = TriangleTests::BarycentricCoordinate(position, vertics[tri[0]].position, vertics[tri[1]].position, vertics[tri[2]].position);
		auto uv = XMVectorBaryCentricV(vertics[tri[0]].uv, vertics[tri[1]].uv, vertics[tri[2]].uv, BC);

		// append uv forcely
		m_uvCurve.append(uv, true);
	}
	m_dirty |= 2;
	return false;
}

void SurfacePatch::closeLoop()
{
	m_boundry.closeLoop();
	m_uvCurve.closeLoop();
}

XMVECTOR SurfacePatch::unproject(XMVECTOR uv, int& fid)
{
	auto& vertics = m_surface->vertices;
	auto& tri = m_surface->facet(fid);
	auto BC = TriangleTests::BarycentricCoordinate(uv, vertics[tri[0]].uv, vertics[tri[1]].uv, vertics[tri[2]].uv);


	XMVECTOR vali = XMVectorLessOrEqual(BC, XMVectorZero());
	XMUINT3 out;
	XMStoreUInt3(&out, vali);

	if (out.x)
		fid = m_surface->adjacentFacet(fid, 0);
	else if (out.y)
		fid = m_surface->adjacentFacet(fid, 1);
	else if (out.z)
		fid = m_surface->adjacentFacet(fid, 2);
	else
		return XMVectorBaryCentricV(vertics[tri[0]].position, vertics[tri[1]].position, vertics[tri[2]].position, BC);
	return unproject(uv, fid);


}

void SurfacePatch::unprojectBoundry(int startFid)
{
	assert(!"Not implemented yet");
	m_dirty = 0;
}

MeshType & SurfacePatch::triangulate(size_t subdiv)
{
	assert(!"Not implemented yet");
	m_dirty = 0;
	return m_mesh;
}
