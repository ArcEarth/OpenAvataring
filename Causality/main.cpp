// Source.cpp : Defines the entry point for the console application.
//
#include "pch_bcl.h"
#include "CausalityApplication.h"
//#include <fbxsdk.h>

using namespace std;
using namespace Leap;
using namespace Causality;
using namespace DirectX;

#include <Windows.h>
#include <string>
#include <vector>

#if defined(__cplusplus_winrt)
using namespace Platform;
using namespace Windows::Globalization;
using namespace Windows::ApplicationModel;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::UI::Core;
using namespace Windows::UI::Input;
using namespace Windows::System;
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;

[Platform::MTAThread]
int WinMain(Platform::Array<Platform::String^>^ args)
#else
int CALLBACK WinMain(
	_In_ HINSTANCE hInstance,
	_In_ HINSTANCE hPrevInstance,
	_In_ LPSTR     lpCmdLine,
	_In_ int       nCmdShow
	)
#endif
{
	std::vector<std::string> args;
	//for (size_t i = 0; i < argc; i++)
	//{
	//	args[i] = argv[i];
	//}

	return Application::Invoke<Causality::App>(args);

	//Leap::Controller controller;
	//SampleListener listener;
	//controller.addListener(listener);

	//window = make_shared<Platform::NativeWindow>();
	//window->Initialize(ref new String(L"Causality"), 1280U, 720,false);
	//deviceResources = make_shared<DirectX::DeviceResources>();
	//deviceResources->SetNativeWindow(window->Handle());
	//auto pRift = std::make_shared<Platform::Devices::OculusRift>();
	//auto pPlayer = std::make_unique<Player>();

	//try
	//{
	//	pRift->Initialize(window->Handle(), deviceResources.get());
	//	pRift->DissmisHealthWarnning();
	//}
	//catch (std::runtime_error exception)
	//{
	//	pRift = nullptr;
	//}

	//if (pRift)
	//{
	//	pPlayer->EnableStereo(pRift);
	//}

	//m_main = make_unique<Causality::DXAppMain>(deviceResources);

	//pPlayer->SetPosition(Fundation::Vector3(0.0f, 0.7f, 1.5f));
	//pPlayer->FocusAt(Fundation::Vector3(0, 0, 0), Fundation::Vector3(0.0f, 1.0f, 0));
	//auto size = deviceResources->GetOutputSize();
	//pPlayer->SetFov(75.f*XM_PI / 180.f, size.Width / size.Height);
	//MSG msg;
	//bool done = false;
	//while (!done)
	//{
	//	// Handle the windows messages.
	//	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	//	{
	//		TranslateMessage(&msg);
	//		DispatchMessage(&msg);
	//	}

	//	// If windows signals to end the application then exit out.
	//	if (msg.message == WM_QUIT)
	//	{
	//		done = true;
	//	}
	//	else
	//	{
	//		m_main->Update();

	//		if (pRift)
	//		{
	//			pRift->BeginFrame();
	//			// Otherwise do the frame processing.
	//			for (int eye = 0; eye < 2; eye++)
	//			{
	//				pRift->EyeTexture((DirectX::Scene::EyesEnum) eye).SetAsRenderTarget(deviceResources->GetD3DDeviceContext(), pRift->DepthStencilBuffer());

	//				auto view = pPlayer->GetViewMatrix((DirectX::Scene::EyesEnum) eye);
	//				auto projection = pPlayer->GetProjectionMatrix((DirectX::Scene::EyesEnum) eye);

	//				m_main->m_sceneRenderer->UpdateViewMatrix(view);
	//				m_main->m_sceneRenderer->UpdateProjectionMatrix(projection);
	//				m_main->m_pSkyBox->UpdateViewMatrix(view);
	//				m_main->m_sceneRenderer->UpdateProjectionMatrix(projection);
	//				m_main->Render();
	//			}
	//			pRift->EndFrame();
	//		}
	//		else
	//		{
	//			auto context = deviceResources->GetD3DDeviceContext();

	//			// Reset the viewport to target the whole screen.
	//			auto viewport = deviceResources->GetScreenViewport();
	//			context->RSSetViewports(1, &viewport);
	//			ID3D11RenderTargetView *const targets[1] = { deviceResources->GetBackBufferRenderTargetView() };
	//			context->OMSetRenderTargets(1, targets, deviceResources->GetDepthStencilView());
	//			context->ClearRenderTargetView(deviceResources->GetBackBufferRenderTargetView(), DirectX::Colors::White);
	//			context->ClearDepthStencilView(deviceResources->GetDepthStencilView(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
	//	
	//			auto view = pPlayer->GetViewMatrix();
	//			auto projection = pPlayer->GetProjectionMatrix();
	//			m_main->m_sceneRenderer->UpdateViewMatrix(view);
	//			m_main->m_sceneRenderer->UpdateProjectionMatrix(projection);
	//			m_main->m_sceneRenderer->Render(context);
	//			m_main->m_pSkyBox->UpdateViewMatrix(view);
	//			m_main->m_pSkyBox->UpdateProjectionMatrix(projection);
	//			m_main->m_pSkyBox->Render(context);
	//			//m_main->Render();
	//			deviceResources->Present();
	//		}
	//	}
	//}

	////std::cin.get();
	////auto calendar = ref new Calendar;
	////calendar->SetToNow();
	////wcout << "It's now " << calendar->HourAsPaddedString(2)->Data() << L":" <<
	////	calendar->MinuteAsPaddedString(2)->Data() << L":" <<
	////	calendar->SecondAsPaddedString(2)->Data() << endl;
	////Platform::Details::Console::WriteLine("Hello World");

	//controller.removeListener(listener);
	//system("Pause");
	return 0;
}

//void OnActivated(Causality::Window ^sender, Windows::UI::Core::WindowActivatedEventArgs ^args)
//{
//	Platform::Details::Console::WriteLine("Lalalaa Demacia!");
//}
