#include "pch_bcl.h"
#include "VisualObject.h"
#include <Models.h>
#include <PrimitiveVisualizer.h>
#include <ShadowMapGenerationEffect.h>
#include "Scene.h"
#include "AssetDictionary.h"

using namespace Causality;
using namespace DirectX;
using namespace DirectX::Visualizers;

namespace Causality
{
	extern bool		g_DebugView;
	extern bool		g_ShowCharacterMesh;
	extern float	g_DebugArmatureThinkness;
	extern bool		g_MirrowInputX;
}

REGISTER_SCENE_OBJECT_IN_PARSER(object, VisualObject);
REGISTER_SCENE_OBJECT_IN_PARSER(glowing_border, GlowingBorder);
REGISTER_SCENE_OBJECT_IN_PARSER(coordinate_axis, CoordinateAxis);

void XM_CALLCONV DrawBox(_In_reads_(8) Vector3 *conners, FXMVECTOR color)
{
	auto& drawer = Visualizers::g_PrimitiveDrawer;
	drawer.DrawLine(conners[0], conners[1], color);
	drawer.DrawLine(conners[1], conners[2], color);
	drawer.DrawLine(conners[2], conners[3], color);
	drawer.DrawLine(conners[3], conners[0], color);

	drawer.DrawLine(conners[0], conners[4], color);
	drawer.DrawLine(conners[1], conners[5], color);
	drawer.DrawLine(conners[2], conners[6], color);
	drawer.DrawLine(conners[3], conners[7], color);

	drawer.DrawLine(conners[4], conners[5], color);
	drawer.DrawLine(conners[5], conners[6], color);
	drawer.DrawLine(conners[6], conners[7], color);
	drawer.DrawLine(conners[7], conners[4], color);

}

void XM_CALLCONV Causality::DrawGeometryOutline(const BoundingGeometry& geometry, FXMVECTOR color)
{
	Vector3 conners[8];
	if (geometry.Type == BoundingGeometryType::Geometry_Frustum)
	{
		geometry.Frustum.GetCorners(conners);
		DrawBox(conners, color);
	}
	else if (geometry.Type == BoundingGeometryType::Geometry_OrientedBox)
	{
		geometry.OrientedBox.GetCorners(conners);
		DrawBox(conners, color);
	}
	else if (geometry.Type == BoundingGeometryType::Geometry_AxisAlignedBox)
	{
		geometry.AxisAlignedBox.GetCorners(conners);
		DrawBox(conners, color);
	}
	else if (geometry.Type == BoundingGeometryType::Geometry_Sphere)
	{
	}
}


IModelNode * VisualObject::RenderModel(int LoD)
{
	return m_pRenderModel;
}

const IModelNode * VisualObject::RenderModel(int LoD) const
{
	return m_pRenderModel;
}

void VisualObject::SetRenderModel(IModelNode * pMesh, int LoD)
{
	m_pRenderModel = pMesh;
}

RenderFlags VisualObject::GetRenderFlags() const
{
	return RenderFlags::OpaqueObjects;
}

void VisualObject::Render(IRenderContext * pContext, IEffect* pEffect)
{
	auto& drawer = g_PrimitiveDrawer;
	if (g_ShowCharacterMesh && m_pRenderModel)
	{
		pContext->RSSetState(drawer.GetStates()->CullNone());
		m_pRenderModel->Render(pContext, GlobalTransformMatrix(), pEffect);
	}

	if (g_ShowCharacterMesh && g_DebugView && m_pRenderModel)
	{
		BoundingGeometry geo(m_pRenderModel->GetBoundingBox());
		geo.Transform(geo, GlobalTransformMatrix());
		drawer.Begin();
		DrawGeometryOutline(geo, Colors::Orange);
		drawer.End();
	}
}

void XM_CALLCONV VisualObject::UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection)
{
}

VisualObject::VisualObject()
{
	m_isVisable = true;
	m_opticity = 1.0f;
	m_isFocuesd = false;
}

Causality::VisualObject::~VisualObject()
{

}

void VisualObject::Parse(const ParamArchive* store)
{
	SceneObject::Parse(store);

	GetParam(store, "visible", m_isVisable);

	float mass = 1.0f;
	GetParam(store, "mass", mass);

	auto& m_assets = Scene->Assets();
	const char* path = nullptr;
	GetParam(store, "mesh", path);
	if (path != nullptr && strlen(path) != 0)
	{
		if (path[0] == '{') // asset reference
		{
			const std::string key(path + 1, path + strlen(path) - 1);
			SetRenderModel(m_assets.GetMesh(key));
		}
	}
	else
	{
		auto nMesh = GetFirstChildArchive(store, "object.mesh");
		if (nMesh)
		{
			nMesh = GetFirstChildArchive(nMesh);
			auto model = m_assets.ParseMesh(nMesh);
			SetRenderModel(model);
		}
	}
}

bool VisualObject::IsVisible(const BoundingGeometry & viewFrustum) const
{
	if (!m_isVisable || m_pRenderModel == nullptr) return false;
	auto box = m_pRenderModel->GetOrientedBoundingBox();
	box.Transform(box, this->GlobalTransformMatrix());
	return viewFrustum.Contains(box) != ContainmentType::DISJOINT;
}


CoordinateAxis::CoordinateAxis()
{
	m_max = 1.0f;
	m_majorIdent = 1.0f;
	m_minorIdent = 0.25f;
	m_ZX = false;
	m_YZ = false;
	m_XY = false;
}

CoordinateAxis::~CoordinateAxis()
{
}

void CoordinateAxis::Parse(const ParamArchive * archive)
{
	GetParam(archive, "max", m_max);
	GetParam(archive, "major_ident", m_majorIdent);
	GetParam(archive, "minor_ident", m_minorIdent);
	GetParam(archive, "zx", m_ZX);
	GetParam(archive, "xy", m_XY);
	GetParam(archive, "yz", m_YZ);
}

bool CoordinateAxis::IsVisible(const BoundingGeometry & viewFrustum) const
{
	return g_DebugView;
}

void CoordinateAxis::Render(IRenderContext * context, IEffect* pEffect)
{
	float ub = m_max, lb = -m_max, majorIdent = m_majorIdent, minorIdent = m_minorIdent;
	using Visualizers::g_PrimitiveDrawer;
	using namespace DirectX;
	//g_PrimitiveDrawer.DrawSphere({ .0f,.0f,.0f,0.02f }, Colors::Cyan);
	g_PrimitiveDrawer.SetWorld(GlobalTransformMatrix());

	float Ar = 0.03f, Al = ub, Almr = Al - Ar, Alpr = Al + Ar;
	g_PrimitiveDrawer.Begin();

	if (m_ZX)
	{
		for (float x = lb; x <= ub; x += minorIdent)
		{
			g_PrimitiveDrawer.DrawLine({ -Al,.0f,x }, { Al,.0f,x }, Colors::DarkGray);
			g_PrimitiveDrawer.DrawLine({ x,.0f,-Al }, { x,.0f,Al }, Colors::DarkGray);
		}

		for (float x = lb; x <= ub; x += majorIdent)
		{
			g_PrimitiveDrawer.DrawLine({ -Al,.0f,x }, { Al,.0f,x }, Colors::DimGray);
			g_PrimitiveDrawer.DrawLine({ x,.0f,-Al }, { x,.0f,Al }, Colors::DimGray);
		}
	}

	g_PrimitiveDrawer.DrawLine({ -Al,.0f,.0f }, { Al,.0f,.0f }, Colors::Red);
	g_PrimitiveDrawer.DrawLine({ .0f,-Al,.0f }, { .0f,Al,.0f }, Colors::Lime);
	g_PrimitiveDrawer.DrawLine({ .0f,.0f,-Al }, { .0f,.0f,Al }, Colors::Blue);
	g_PrimitiveDrawer.End();

	g_PrimitiveDrawer.Begin();
	g_PrimitiveDrawer.DrawTriangle({ Alpr,.0f,.0f }, { Almr,Ar,.0f }, { Almr,-Ar,.0f }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ Alpr,.0f,.0f }, { Almr,-Ar,.0f }, { Almr,Ar,.0f }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ Alpr,.0f,.0f }, { Almr,.0f,Ar }, { Almr,.0f,-Ar }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ Alpr,.0f,.0f }, { Almr,.0f,-Ar }, { Almr,.0f,Ar }, Colors::Red);

	g_PrimitiveDrawer.DrawTriangle({ .0f,Alpr,.0f }, { -Ar,Almr,.0f }, { Ar,Almr,.0f }, Colors::Lime);
	g_PrimitiveDrawer.DrawTriangle({ .0f,Alpr,.0f }, { Ar,Almr,.0f }, { -Ar,Almr,.0f }, Colors::Lime);
	g_PrimitiveDrawer.DrawTriangle({ .0f,Alpr,.0f }, { .0f,Almr,-Ar }, { .0f,Almr,Ar }, Colors::Lime);
	g_PrimitiveDrawer.DrawTriangle({ .0f,Alpr,.0f }, { .0f,Almr,Ar }, { .0f,Almr,-Ar }, Colors::Lime);

	g_PrimitiveDrawer.DrawTriangle({ .0f,.0f,Alpr }, { Ar,.0f,Almr }, { -Ar,.0f,Almr }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ .0f,.0f,Alpr }, { -Ar,.0f,Almr }, { Ar,.0f,Almr }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ .0f,.0f,Alpr }, { .0f,Ar,Almr }, { .0f,-Ar,Almr }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ .0f,.0f,Alpr }, { .0f,-Ar,Almr }, { .0f,Ar,Almr }, Colors::Blue);
	g_PrimitiveDrawer.End();
}

void XM_CALLCONV CoordinateAxis::UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection)
{
	using Visualizers::g_PrimitiveDrawer;
	g_PrimitiveDrawer.SetView(view);
	g_PrimitiveDrawer.SetProjection(projection);
}

RenderFlags CoordinateAxis::GetRenderFlags() const
{
	return RenderFlags::SpecialEffects;
}

GlowingBorder::GlowingBorder()
{
	m_Color = Colors::Red.v;
}

GlowingBorder::GlowingBorder(const Color & color)
	: m_Color(color)
{

}

bool GlowingBorder::IsVisible(const BoundingGeometry & viewFrustum) const
{
	if (!IsEnabled()) return false;
	auto pVisual = this->FirstAncesterOfType<VisualObject>();
	return pVisual && pVisual->IsVisible(viewFrustum);
}

RenderFlags GlowingBorder::GetRenderFlags() const
{
	return RenderFlags::BloomEffectSource;
}

void GlowingBorder::Render(IRenderContext * pContext, IEffect * pEffect)
{
	auto pVisual = this->FirstAncesterOfType<VisualObject>();
	auto pModel = pVisual ? pVisual->RenderModel() : nullptr;
	auto pSGEffect = dynamic_cast<ShadowMapGenerationEffect*> (pEffect);
	if (pSGEffect)
	{
		if (pSGEffect->GetShadowFillMode() == ShadowMapGenerationEffect::SolidColorFill)
		{
			pSGEffect->SetShadowColor(m_Color);
		}
	}

	if (pModel)
	{
		pModel->Render(pContext, pVisual->GlobalTransformMatrix(), pEffect); // Render parent model with customized effect
	}
}

void XM_CALLCONV GlowingBorder::UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection)
{
}
