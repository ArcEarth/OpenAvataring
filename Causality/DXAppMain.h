#pragma once

#include "Common\StepTimer.h"
#include "Common\DeviceResources.h"
#include "Content\CubeScene.h"
#include "Content\SampleFpsTextRenderer.h"

// Renders Direct2D and 3D content on the screen.
namespace Causality
{
	class DXAppMain : public DirectX::IDeviceNotify
	{
	public:
		DXAppMain(const std::shared_ptr<DirectX::DeviceResources>& deviceResources);
		~DXAppMain();
		void CreateWindowSizeDependentResources();
		void Update();
		bool Render();

		void StartTracking() { m_sceneRenderer->StartTracking(); }
		void TrackingUpdate(float positionX) { m_pointerLocationX = positionX; }
		void StopTracking() { m_sceneRenderer->StopTracking(); }
		bool IsTracking() { return m_sceneRenderer->IsTracking(); }
		// IDeviceNotify
		virtual void OnDeviceLost();
		virtual void OnDeviceRestored();

	public:
		// Cached pointer to device resources.
		std::shared_ptr<DirectX::DeviceResources> m_deviceResources;

		// TODO: Replace with your own content renderers.
		std::unique_ptr<CubeScene> m_sceneRenderer;
		std::unique_ptr<HUDInterface> m_fpsTextRenderer;
		//std::unique_ptr<DirectX::Scene::SkyDome> m_pSkyBox;

		// Rendering loop timer.
		DirectX::StepTimer m_timer;

		float m_pointerLocationX;
	};
}