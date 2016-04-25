#include "pch_bcl.h"
#include "FloatHud.h"
#include <PrimitiveVisualizer.h>
#include <Textures.h>
#include <Material.h>
#include <DirectXHelper.h>
#include "Scene.h"

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
	m_placement = SpritePlacement_Top;
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

	auto world = this->GetGlobalTransform().TransformMatrix();
	auto worldView = XMMatrixMultiply(world ,m_view);

	auto particle = XMParticleProjection(Vector4(.0f,.0f,.0f,1.0), worldView, m_proj, viewport);
	float depth = _DXMEXT XMVectorGetZ(particle);
	particle = _DXMEXT XMVectorSwizzle<0, 1, 3, 3>(particle);
	

	Vector4 fp = particle;

	m_sizeProj.x = m_pTexture->Width() * m_Transform.GblScaling.x;
	m_sizeProj.y = m_pTexture->Height() * m_Transform.GblScaling.y;

	fp.z = m_sizeProj.x; fp.w = m_sizeProj.y;

	RECT rect;
	rect.left = fp.x - fp.z;
	rect.right = fp.x + fp.z;
	rect.top = fp.y - fp.w;
	rect.bottom = fp.y + fp.w;
	auto srv = m_pTexture->ShaderResourceView();
	auto color = Colors::White;

	auto sprites = g_PrimitiveDrawer.GetSpriteBatch();
	sprites->Begin(SpriteSortMode_Immediate, nullptr, g_PrimitiveDrawer.GetStates()->PointClamp());
	//sprites->Draw(srv,Vector2(fp.x,fp.y));
	sprites->Draw(srv, rect, nullptr, color, 0, Vector2::Zero, DirectX::SpriteEffects::SpriteEffects_None, depth);
	sprites->End();
}

void XM_CALLCONV SpriteObject::UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection)
{
	m_view = view;
	m_proj = projection;

	XMVECTOR wp = GetPosition();
	XMMATRIX vp = view * projection;
	m_positionProj = XMVector3TransformCoord(wp, vp);

	//XMVECTOR viewport = XMLoad(m_viewportSize);
	//XMVector3ConvertToTextureCoord(GetPosition(), viewport, view*projection);
}

I2DContext * Causality::SpriteObject::Get2DContext(IRenderDevice * pDevice)
{
	return this->Scene->Get2DContext();;
}

I2DContext * Causality::SpriteObject::Get2DContext(IRenderContext * pContext)
{
	return this->Scene->Get2DContext();;
}

I2DFactory * Causality::SpriteObject::Get2DFactory()
{
	return this->Scene->Get2DFactory();
}

ITextFactory * Causality::SpriteObject::GetTextFactory()
{
	return this->Scene->GetTextFactory();
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


Causality::SpriteCanvas::SpriteCanvas()
{
}

Causality::SpriteCanvas::~SpriteCanvas()
{
}

SpriteCanvas::SpriteCanvas(IRenderDevice * pDevice, size_t width, size_t height)
{
	CreateDeviceResources(pDevice, width, height);
}

void SpriteCanvas::CreateDeviceResources(IRenderDevice* pDevice, size_t width, size_t height)
{
	using namespace DirectX;
	auto p2DContext = Get2DContext(pDevice);
	m_clearCanvasWhenRender = true;
	m_canvas = CanvasFacade::CreateTarget(pDevice, p2DContext, width, height);

	CanvasFacade::SetTarget(m_canvas.get());
	SpriteObject::SetTexture(m_canvas.get());

	m_decalMat.reset(new PhongMaterial());
	m_decalMat->DiffuseMap = m_canvas->ShaderResourceView();

	CanvasFacade::CreateDeviceResources(Get2DFactory(), GetTextFactory());
}
