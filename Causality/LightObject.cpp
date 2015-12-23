#include "pch_bcl.h"
#include "CameraObject.h"
#include <DirectXColors.h>
#include "LightObject.h"
#include <PrimitiveVisualizer.h>
#include <ShadowMapGenerationEffect.h>
#include "Scene.h"

namespace Causality
{
	extern bool g_DebugView;
}

using namespace DirectX;
using namespace Causality;

REGISTER_SCENE_OBJECT_IN_PARSER(light, Light);

std::weak_ptr<ShadowMapGenerationEffect> g_wpSMGEffect;

Light::Light(IRenderDevice * device, const UINT& shadowResolution)
{
	EnableDropShadow(device, shadowResolution);
}

void Light::Parse(const ParamArchive * store)
{
	using namespace DirectX;
	Camera::Parse(store);
	unsigned resolution = 1024;
	bool enableShadow;
	Color color = Colors::White.v;
	GetParam(store, "color", color);
	GetParam(store, "drops_shadow", enableShadow);
	GetParam(store, "resolution", resolution);

	SetColor(color);
	if (enableShadow)
	{
		auto device = Scene->GetRenderDevice();
		EnableDropShadow(device, resolution);
	}
}

Light::Light()
{
}

void Light::Begin(IRenderContext* context)
{
	m_RenderTarget.Clear(context,m_Background);
	auto pEffect = static_cast<ShadowMapGenerationEffect*>(m_pShadowMapGenerator.get());

	m_RenderTarget.SetAsRenderTarget(context);

	if (pEffect)
	{
		pEffect->SetShadowMap(NULL, NULL);
		pEffect->SetView(GetViewMatrix());
		pEffect->SetProjection(GetProjectionMatrix());
	}
}

void Light::End()
{
}

IEffect * Light::GetRenderEffect()
{
	return m_pShadowMapGenerator.get();
}

bool Light::AcceptRenderFlags(RenderFlags flags)
{
	bool result = flags.Contains(RenderFlags(RenderFlags::OpaqueObjects));
	return result;
}

Color Light::GetColor() const { return m_Color; }

void Light::DisableDropShadow()
{
	m_pShadowMapGenerator.reset();
	m_RenderTarget.Reset();
}

void Light::EnableDropShadow(IRenderDevice * device, const UINT & resolution)
{
	if (!device)
		return;
	if (!g_wpSMGEffect.expired())
		m_pShadowMapGenerator = g_wpSMGEffect.lock();
	else
	{
		auto pEffect = std::make_shared<ShadowMapGenerationEffect>(device);
		g_wpSMGEffect = pEffect;
		m_pShadowMapGenerator = pEffect;
	}

	CD3D11_VIEWPORT viewport(.0f, .0f, resolution, resolution);
	m_RenderTarget = RenderTarget(
		RenderableTexture2D(device, resolution, resolution, DXGI_FORMAT_R16_UNORM),
		DepthStencilBuffer(device, resolution, resolution, DXGI_FORMAT_D16_UNORM),
		viewport);
}

void Light::SetShadowMapBuffer(const DepthStencilBuffer & depthBuffer)
{
	m_RenderTarget.DepthBuffer() = depthBuffer;
}

ID3D11ShaderResourceView * Causality::Light::GetShadowMap() const { 
	return m_RenderTarget.DepthBuffer().ShaderResourceView();
	//return m_RenderTargetTex.ShaderResourceView();
}

ID3D11ShaderResourceView * Causality::Light::GetColorMap() const {
	return m_pColorMap.Get();
}


// Inherited via IVisual

RenderFlags Light::GetRenderFlags() const { return RenderFlags::Lights | RenderFlags::SpecialEffects; }

bool Light::IsVisible(const BoundingGeometry & viewFrustum) const
{
	return viewFrustum.Contains(this->GetViewFrustum()) != ContainmentType::DISJOINT;
}

void Light::Render(IRenderContext * context, DirectX::IEffect* pEffect)
{
	if (g_DebugView)
	{
		auto &geometry = GetViewFrustum();
		auto& drawer = Visualizers::g_PrimitiveDrawer;
		drawer.SetWorld(XMMatrixIdentity());
		drawer.Begin();
		DrawGeometryOutline(geometry, m_Color);
		drawer.End();
	}
}

void XM_CALLCONV Light::UpdateViewMatrix(FXMMATRIX view, CXMMATRIX projection)
{
	auto& drawer = Visualizers::g_PrimitiveDrawer;
	drawer.SetView(view);
	drawer.SetProjection(projection);
}

DirectX::XMVECTOR Causality::ILight::GetFocusDirection() const
{
	using namespace DirectX;
	XMMATRIX view = AsViewControl()->GetViewMatrix();
	view = XMMatrixTranspose(view);
	return XMVectorSelect(g_XMSelect1110.v, -view.r[2], g_XMSelect1110.v);
}
