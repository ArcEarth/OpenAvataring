#include "pch_bcl.h"
#include "SkyDome.h"
#include <Models.h>
#include <PrimitiveVisualizer.h>
#include "AssetDictionary.h"
#include "Scene.h"
#include <SkyDomeEffect.h>

using namespace Causality;
using namespace DirectX;

namespace Causality
{
	extern bool		g_ShowCharacterMesh;
}

REGISTER_SCENE_OBJECT_IN_PARSER(skydome, SkyDome);

SkyDome::SkyDome()
{
}

SkyDome::~SkyDome()
{

}

void SkyDome::Parse(const ParamArchive * store)
{
	SceneObject::Parse(store);
	const char* bg = nullptr;
	GetParam(store, "background", bg);

	auto& m_assets = Scene->Assets();
	
	auto pEffect = m_assets.GetEffect("skydome_effect");
	auto pSeffect = dynamic_cast<DirectX::SkydomeEffect*>(pEffect);

	CreateDeviceResource(m_assets.GetRenderDevice(), pSeffect);
	if (m_pEffect)
	{
		m_assets.AddEffect("skydome_effect", m_pEffect);
	}

	if (bg[0] == '{')
	{
		string key(bg + 1, strlen(bg) - 2);
		SetTexture(*m_assets.GetTexture(key));
	}
	else
	{
		SetTexture(*m_assets.LoadTexture(Name + "_background", bg));
	}
}

void SkyDome::CreateDeviceResource(ID3D11Device * device, DirectX::SkydomeEffect * pEffect)
{
	m_pSphere = DirectX::Scene::GeometricPrimtives::CreateSphere(device, 1.0f, 16, true, true);

	if (pEffect == nullptr)
	{
		pEffect = new SkydomeEffect(device);
	}

	m_pSphere->CreateInputLayout(device, pEffect);
	m_pEffect = pEffect;
}


void SkyDome::SetTexture(DirectX::Texture & texture)
{
	m_Texture = texture;
}

// Inherited via IVisual

bool SkyDome::IsVisible(const DirectX::BoundingGeometry & viewFrustum) const
{
	return g_ShowCharacterMesh;
}

void SkyDome::Render(IRenderContext * context, DirectX::IEffect * pEffect)
{
	auto pStates = Visualizers::g_PrimitiveDrawer.GetStates();
	m_pEffect->SetEnviromentMap(m_Texture.ShaderResourceView());
	m_pEffect->SetWorld(XMMatrixIdentity());
	m_pEffect->Apply(context);

	ComPtr<ID3D11DepthStencilState> pFomerState;
	UINT sRef;
	context->RSSetState(pStates->CullClockwise());
	context->OMGetDepthStencilState(&pFomerState, &sRef);
	context->OMSetDepthStencilState(pStates->DepthNone(), sRef);

	m_pSphere->Draw(context, m_pEffect);

	context->OMSetDepthStencilState(pFomerState.Get(), sRef);
	context->RSSetState(pStates->CullCounterClockwise());
}

void XM_CALLCONV SkyDome::UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection)
{
	XMMATRIX View = view;
	// Last column of View Inverse is camera's position
	View.r[3] = g_XMIdentityR3;
	m_pEffect->SetView(View);
	View = projection;
	m_pEffect->SetProjection(View);
}

// Inherited via IVisual

RenderFlags SkyDome::GetRenderFlags() const
{
	return RenderFlags::SpecialEffects;
}
