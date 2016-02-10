#include "pch_bcl.h"
#include "FloatHud.h"
#include <PrimitiveVisualizer.h>
#include <Textures.h>
#include <HUD.h>

using namespace Causality;
using namespace DirectX;
using namespace DirectX::Visualizers;

//FloatingHud::FloatingHud(HUDElement * hudElem)
//{
//
//}
//
//void FloatingHud::Update(time_seconds const & time_delta)
//{
//	SceneObject::Update(time_delta);
//}

SpriteObject::SpriteObject()
{
	m_pTexture = nullptr;
}

bool SpriteObject::IsVisible(const BoundingGeometry & viewFrustum) const
{
	auto pVisual = FirstAncesterOfType<IVisual>();
	return m_pTexture != nullptr && pVisual && pVisual->IsVisible(viewFrustum);
	//BoundingGeometry geo;
	//GetBoundingGeometry(geo);
	//return  && viewFrustum.Contains(geo);
}

RenderFlags SpriteObject::GetRenderFlags() const
{
	return RenderFlags::SpecialEffects;
}

void SpriteObject::Render(IRenderContext * pContext, IEffect * pEffect)
{
	if (!m_pTexture)
		return;
	UINT numViewport = 1;
	D3D11_VIEWPORT viewport;
	pContext->RSGetViewports(&numViewport, &viewport);

	XMVECTOR vp = XMLoadFloat4(&viewport.TopLeftX);
	XMVECTOR sp = XMVector3ConvertToTextureCoord(XMLoad(m_positionProj), vp);
	Vector2 cen = sp;
	Vector2 sz = m_sizeProj / 2;
	sp = XMVectorZero();

	auto sprites = g_PrimitiveDrawer.GetSpriteBatch();
	sprites->Begin(SpriteSortMode_Immediate,nullptr,g_PrimitiveDrawer.GetStates()->PointClamp());
	RECT trect;
	trect.left = cen.x - sz.x;
	trect.right = cen.x + sz.x;
	trect.top = cen.y - sz.y;
	trect.bottom = cen.y + sz.y;
	sprites->Draw(m_pTexture->ShaderResourceView(), sp,nullptr,Colors::White,0,Vector2::Zero,m_Transform.GblScaling,SpriteEffects_None,m_positionProj.z);
	sprites->End();
}

void XM_CALLCONV SpriteObject::UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection)
{
	XMVECTOR wp = GetPosition();
	XMMATRIX vp = view * projection;
	m_positionProj = XMVector3TransformCoord(wp, vp);

	//XMVECTOR viewport = XMLoad(m_viewportSize);
	//XMVector3ConvertToTextureCoord(GetPosition(), viewport, view*projection);
}

//XMVECTOR SpriteObject::GetSize() const
//{
//	return GetScale();
//}
//
//void SpriteObject::SetSize(FXMVECTOR size)
//{
//	Vector3 s = size;
//	s.z = 0.001f;
//	SetScale(s);
//}

bool SpriteObject::GetBoundingBox(BoundingBox & box) const
{
	auto transform = GetGlobalTransform().TransformMatrix();
	BoundingBox tbox;
	tbox.Extents = GetScale();
	tbox.Transform(box, transform);
	return true;
}

bool SpriteObject::GetBoundingGeometry(BoundingGeometry & geo) const
{
	auto transform = GetGlobalTransform().TransformMatrix();
	BoundingGeometry tgeo;
	tgeo.AxisAlignedBox = BoundingBox();
	tgeo.AxisAlignedBox.Extents = GetScale();
	tgeo.Transform(geo, transform);
	return false;
}

Texture2D * SpriteObject::GetTexture() const
{
	return m_pTexture;
}

void SpriteObject::SetTexture(Texture2D * texture)
{
	m_pTexture = texture;
}
