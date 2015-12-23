#pragma once

#include "..\Common\DeviceResources.h"
#include "ShaderStructures.h"
#include "..\Common\StepTimer.h"
#include "..\Common\Renderable.h"
#include "..\Interactive.h"

namespace Causality
{
	// This sample renderer instantiates a basic rendering pipeline.
	class CubeScene : public IAppComponent, public DirectX::Scene::IRenderable, public DirectX::Scene::ITimeAnimatable, public DirectX::Scene::IViewable, public ICursorInteractive, public IUserHandsInteractive
	{
	public:
		CubeScene(const std::shared_ptr<DirectX::DeviceResources>& deviceResources);
		void CreateDeviceDependentResources();
		void CreateWindowSizeDependentResources();
		void ReleaseDeviceDependentResources();

		// ITimeAnimatable
		void UpdateAnimation(DirectX::StepTimer const& timer) override;
		// IRenderable
		void Render(ID3D11DeviceContext *pContext) override;

		// IViewable
		void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override;
		//void XM_CALLCONV UpdateProjectionMatrix(DirectX::FXMMATRIX projection) override;


		void StartTracking();
		void TrackingUpdate(float positionX);
		void StopTracking();
		bool IsTracking() { return m_tracking; }

		// Inherited via ICursorInteractive
		virtual void OnMouseButtonDown(const CursorButtonEvent & e) override;
		virtual void OnMouseButtonUp(const CursorButtonEvent & e) override;
		virtual void OnMouseMove(const CursorMoveEventArgs & e) override;

		// Inherited via IUserHandsInteractive
		virtual void OnHandsTracked(const UserHandsEventArgs & e) override;
		virtual void OnHandsTrackLost(const UserHandsEventArgs & e) override;
		virtual void OnHandsMove(const UserHandsEventArgs & e) override;

	private:
		void Rotate(float radians);

	private:
		// Cached pointer to device resources.
		std::shared_ptr<DirectX::DeviceResources> m_deviceResources;

		// Direct3D resources for cube geometry.
		Microsoft::WRL::ComPtr<ID3D11InputLayout>	m_inputLayout;
		Microsoft::WRL::ComPtr<ID3D11Buffer>		m_vertexBuffer;
		Microsoft::WRL::ComPtr<ID3D11Buffer>		m_indexBuffer;
		Microsoft::WRL::ComPtr<ID3D11VertexShader>	m_vertexShader;
		Microsoft::WRL::ComPtr<ID3D11PixelShader>	m_pixelShader;
		Microsoft::WRL::ComPtr<ID3D11Buffer>		m_constantBuffer;

		// System resources for cube geometry.
		ModelViewProjectionConstantBuffer	m_constantBufferData;
		uint32_t	m_indexCount;

		// Variables used with the rendering loop.
		bool	m_loadingComplete;
		float	m_degreesPerSecond;
		bool	m_tracking;

	};
}

