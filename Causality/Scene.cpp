#include "pch_bcl.h"
#include "Scene.h"
#include <ShaderEffect.h>
#include "AssetDictionary.h"
#include "CameraObject.h"
#include "LightObject.h"
#include "VisualObject.h"
#include <HUD.h>
#include <tinyxml2.h>

using namespace Causality;
using namespace std;

Scene::Scene()
{
	time_scale = 1.0f;
	is_paused = false;
	is_loaded = false;
	m_primCamera = nullptr;
	m_assets = make_unique<AssetDictionary>();
}

Scene::~Scene()
{
	int* p = nullptr;
}

// Contens operations

SceneObject * Scene::Content() { return m_sceneRoot.get(); }

SceneObject * Scene::SetContent(SceneObject * sceneRoot) { m_sceneRoot.reset(sceneRoot); return sceneRoot; }

inline HUDCanvas * Causality::Scene::GetHudCanvas() { return m_hudRoot.get(); }

inline void Causality::Scene::SetHudCanvas(HUDCanvas * canvas) { m_hudRoot.reset(canvas); }

ICamera * Scene::PrimaryCamera()
{
	return m_primCamera;
}

bool Scene::SetAsPrimaryCamera(ICamera * camera)
{
	//if (camera->Scene != this) return false;
	m_primCamera = camera;
	//camera->SetRenderTarget(Canvas());
	return true;
}

void Scene::SetRenderDeviceAndContext(IRenderDevice * device, IRenderContext * context)
{
	m_assets->SetRenderDevice(device);
	m_device = device;
	m_context = context;
}

void Scene::SetHudRenderDevice(I2DFactory * pD2dFactory, I2DContext* pD2dContext, ITextFactory * pTextFactory)
{
	m_2dFactory = pD2dFactory;
	m_textFactory = pTextFactory;
	m_2dContext = pD2dContext;
}

void Scene::SetCanvas(DirectX::RenderTarget & canvas) {
	for (auto& camera : m_cameras)
	{
	}
	m_canvas = canvas;
	//back_buffer = canvas.ColorBuffer();
	//m_canvas = DirectX::RenderTarget(DirectX::RenderableTexture2D(render_device, back_buffer.Width(), back_buffer.Height(), back_buffer.Format()), canvas.DepthBuffer());
}

void Scene::UpdateRenderViewCache()
{
	if (!camera_dirty) return;
	m_cameras.clear();
	m_renderables.clear();
	m_lights.clear();
	m_effects.clear();

	auto& aseffects = m_assets->GetEffects();
	for (auto pe : aseffects)
		m_effects.push_back(pe);

	for (auto& obj : m_sceneRoot->nodes())
	{
		auto pCamera = obj.As<ICamera>();
		if (pCamera != nullptr)
		{
			m_cameras.push_back(pCamera);

			//  Register camera render effects
			for (size_t i = 0; i < pCamera->ViewCount(); i++)
			{
				for (size_t j = 0; j < pCamera->ViewRendererCount(i); j++)
				{
					auto pRender = pCamera->GetViewRenderer(i, j);
					auto pEffect = pRender->GetRenderEffect();
					if (pRender != nullptr && std::find(m_effects.begin(), m_effects.end(), pEffect) == m_effects.end())
						m_effects.push_back(pEffect);
				}
			}
		}

		auto pLight = obj.As<ILight>();
		if (pLight != nullptr)
			m_lights.push_back(pLight);

		auto pRenderable = obj.As<IVisual>();
		if (pRenderable != nullptr)
			m_renderables.push_back(pRenderable);
	}
	camera_dirty = false;
}


void Scene::Update()
{

	if (is_paused) return;
	lock_guard<mutex> guard(content_mutex);
	m_timer.Tick([this]() {
		time_seconds deltaTime(m_timer.GetElapsedSeconds() * time_scale);
		for (auto& pObj : m_sceneRoot->nodes())
		{
			if (pObj.IsEnabled())
				pObj.Update(deltaTime);
		}
	});
	if (camera_dirty)
		UpdateRenderViewCache();
}

void Scene::SignalCameraCache() { camera_dirty = true; }

std::mutex & Scene::ContentMutex() { return content_mutex; }

void Scene::Render(IRenderContext * context)
{
	// if (!is_loaded) return;
	lock_guard<mutex> guard(content_mutex);

	SetupEffectsLights(nullptr);

	std::vector<IVisual*> visibles;
	visibles.reserve(m_renderables.size());

	// Render 3D Scene
	for (auto pCamera : m_cameras) // cameras
	{
		auto viewCount = pCamera->ViewCount();

		for (int view = 0; view < viewCount; view++) // camera viewports
		{
			auto pView = pCamera->GetView(view);
			auto v = pView->GetViewMatrix();
			auto p = pView->GetProjectionMatrix();

			auto& viewFrustum = pView->GetViewFrustum();

			auto passCount = pCamera->ViewRendererCount(view);

			visibles.clear();
			for (auto pRenderable : m_renderables)
			{
				if (pRenderable->IsVisible(viewFrustum))
					visibles.push_back(pRenderable);
			}

			for (size_t pass = 0; pass < passCount; pass++) // render passes
			{
				auto pRenderer = pCamera->GetViewRenderer(view, pass);
				auto pEffect = pRenderer->GetRenderEffect();
				SetupEffectsViewProject(pEffect, v, p);
				pRenderer->Begin(context);
				for (auto& pVisible : visibles) // visible objects
				{
					if (pRenderer->AcceptRenderFlags(pVisible->GetRenderFlags()))
					{
						pVisible->UpdateViewMatrix(v, p);
						pVisible->Render(context, pEffect);
					}
				}
				pRenderer->End();
			}
		}
	}

	if (m_hudRoot)
		m_hudRoot->Render(m_2dContext.Get());
	//context->CopyResource(back_buffer.Resource(), m_canvas.ColorBuffer().Resource());
}

vector<DirectX::IEffect*>& Scene::GetEffects() {
	return m_effects;
}

void Scene::SetupEffectsViewProject(DirectX::IEffect* pEffect, const DirectX::XMMATRIX &v, const DirectX::XMMATRIX &p)
{
	auto pME = dynamic_cast<DirectX::IEffectMatrices*>(pEffect);
	if (pME)
	{
		pME->SetView(v);
		pME->SetProjection(p);
	}

	for (auto pEff : m_effects)
	{
		pME = dynamic_cast<DirectX::IEffectMatrices*>(pEff);
		if (pME)
		{
			pME->SetView(v);
			pME->SetProjection(p);
		}
	}
}

void Scene::SetupEffectsLights(DirectX::IEffect * pEffect)
{
	using namespace DirectX;

	XM_ALIGNATTR
	struct LightParam
	{
		XMVECTOR color;
		XMVECTOR direction;
		XMMATRIX view;
		XMMATRIX proj;
		float	 bias;
		ID3D11ShaderResourceView* shadow;
	};

	XMVECTOR ambient = XMVectorSet(0.5f, 0.5f, 0.5f, 1.0f);

	static const int MaxLights = IEffectLights::MaxDirectionalLights;
	LightParam Lps[MaxLights];

	for (int i = 0; i < std::min((int)m_lights.size(), MaxLights); i++)
	{
		auto pLight = m_lights[i];
		Lps[i].color = pLight->GetColor();
		Lps[i].direction = pLight->GetFocusDirection();
		Lps[i].shadow = pLight->GetShadowMap();
		auto pView = pLight->AsViewControl();
		if (pView)
		{
			Lps[i].view = pView->GetViewMatrix();
			Lps[i].proj = pView->GetProjectionMatrix();
			Lps[i].bias = 0.0005f;
		}
	}

	for (auto pEff : m_effects)
	{
		auto pELS = dynamic_cast<DirectX::IEffectLightsShadow*>(pEff);
		auto pEL = dynamic_cast<DirectX::IEffectLights*>(pEff);

		if (pEL)
		{
			pEL->SetAmbientLightColor(ambient);

			for (int i = 0; i < std::min((int)m_lights.size(), MaxLights); i++)
			{
				auto pLight = m_lights[i];
				pEL->SetLightEnabled(i, true);
				pEL->SetLightDiffuseColor(i, Lps[i].color);
				pEL->SetLightSpecularColor(i, Lps[i].color);
				pEL->SetLightDirection(i, Lps[i].direction);

				if (pELS != nullptr)
				{
					pELS->SetLightShadowMap(i, Lps[i].shadow);
					pELS->SetLightView(i, Lps[i].view);
					pELS->SetLightProjection(i, Lps[i].proj);
					pELS->SetLightShadowMapBias(i, Lps[i].bias);
				}

			}
		}
	}
}

void Scene::Load()
{
}

void Scene::Release()
{
}

bool Scene::IsLoading() const
{
	return false;
}

bool Scene::IsReleasing() const
{
	return false;
}

bool Scene::IsLoaded() const
{
	return false;
}

void Scene::OnNavigatedTo()
{
}

void Scene::OnNavigatedFrom()
{
}
