#include "pch_bcl.h"
#include "CausalityApplication.h"
//#include "Content\CubeScene.h"
//#include "Content\SampleFpsTextRenderer.h"
#include <CommonStates.h>
#include <PrimitiveVisualizer.h>
#include <ppltasks.h>

#include "Kinect.h"
#include "OculusRift.h"
#include "LeapMotion.h"
#include "Vicon.h"

#include <tinyxml2.h>

using namespace Causality;
using namespace std;
namespace sys = std::tr2::sys;
using namespace DirectX;
using namespace DirectX::Scene;

string  g_AppManifest = "App.xml";

//std::wstring sceneFile = L"SelectorScene.xml";


IAppComponent::~IAppComponent()
{
	Unregister();
}

void IAppComponent::Register()
{
	App::Current()->RegisterComponent(this);
}

void IAppComponent::Unregister()
{
	App::Current()->UnregisterComponent(this);
}

App::App()
{
}

App::~App()
{
	for (auto& pCom : Components)
	{
		UnregisterComponent(pCom.get());
	}
}

Application::Application()
{
	hInstance = GetModuleHandle(NULL);
}

Application::~Application()
{
}

int Application::Run(const std::vector<std::string>& args)
{
	if (OnStartup(args))
	{
		while (!exitProposal)
		{
			MSG msg;
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			else
			{
				bool succ = OnIdle();
				if (!succ)
					exitProposal = true;
			}
		}
	}

	OnExit();
	return S_OK;
}

void Application::Exit()
{
	for (auto& p : WindowsLookup)
	{
		auto pWindow = p.second.lock();
		if (pWindow)
			pWindow->Close();
	}
	PostQuitMessage(0);
}

bool App::OnStartup(const std::vector<std::string>& args)
{
	using namespace tinyxml2;
	tinyxml2::XMLDocument appDoc(true, Whitespace::COLLAPSE_WHITESPACE);
	auto error = appDoc.LoadFile(g_AppManifest.c_str());
	if (error != XML_SUCCESS)
	{
		string message = "Failed to load or parse file : " + g_AppManifest;
		MessageBoxA(NULL, message.c_str(), "Startup failed", MB_OK);
		return false;
	}

	auto appSettings = appDoc.FirstChildElement("application");
	if (!appSettings)
	{
		MessageBoxA(NULL, "App.xml is not formated correct", "Startup failed", MB_OK);
		return false;
	}

	auto windowSettings = appSettings->FirstChildElement("window");
	auto consoleSettings = appSettings->FirstChildElement("console");

	string assetDir;
	GetParam(appSettings->FirstChildElement("assets"), "path", assetDir);
	string scenestr;
	GetParam(appSettings->FirstChildElement("scenes"), "path", scenestr);

	m_assetsDir = assetDir;
	path sceneFile = scenestr;

	if (m_assetsDir.empty())
		m_assetsDir = sys::current_path();
	else if (m_assetsDir.is_relative())
		m_assetsDir = sys::current_path() / m_assetsDir;

	if (sceneFile.is_relative())
		sceneFile = m_assetsDir / sceneFile;
	if (!sys::exists(sceneFile))
	{
		string message = "Secen file doest exist : " + sceneFile.string();
		MessageBoxA(NULL, message.c_str(), "Startup failed", MB_OK);
		return false;
	}


	string title = "No title";
	GetParam(appSettings, "title", title);

	unsigned width = 1280, height = 720;
	bool fullscreen = false;

	// Initialize Windows
	if (consoleSettings)
	{
		GetParam(consoleSettings, "width", width);
		GetParam(consoleSettings, "height", height);
		GetParam(consoleSettings, "fullscreen", fullscreen);

		pConsole = make_shared<DebugConsole>();
		pConsole->Initialize(title, width, height, fullscreen);
	}

	if (windowSettings)
	{
		GetParam(windowSettings, "width", width);
		GetParam(windowSettings, "height", height);
		GetParam(windowSettings, "fullscreen", fullscreen);
	}

	pWindow = make_shared<NativeWindow>();
	if (!pRift)
		pWindow->Initialize(title, width, height, fullscreen);
	else
	{
		//auto res = pRift->Resoulution();
		Vector2 res = { 1920, 1080 };
		pWindow->Initialize(std::string(title), (unsigned)res.x, (unsigned)res.y, false);
	}
	//bool useOvr = Devices::OculusRift::Initialize();

	// Initialize DirectX
	pDeviceResources = make_shared<DirectX::DeviceResources>();
	pDeviceResources->SetNativeWindow(pWindow->Handle());
	// Register to be notified if the Device is lost or recreated
	pDeviceResources->RegisterDeviceNotify(this);
	pWindow->SizeChanged += MakeEventHandler(&App::OnResize, this);

	//return;

	pDeviceResources->GetD3DDevice()->AddRef();
	pDeviceResources->GetD3DDeviceContext()->AddRef();
	pDevice.Attach(pDeviceResources->GetD3DDevice());
	pContext.Attach(pDeviceResources->GetD3DDeviceContext());

	Visualizers::g_PrimitiveDrawer.Initialize(pContext.Get());

	// Oculus Rift
	//if (pRift)
	//{
	//	if (!pRift->InitializeGraphics(pWindow->Handle(), pDeviceResources.get()))
	//		pRift = nullptr;
	//}

	SetupDevices(appSettings);
	//auto loadingScene = new Scene;
	//Scenes.emplace_back(loadingScene);
	//loadingScene->SetRenderDeviceAndContext(pDevice, pContext);
	//loadingScene->SetCanvas(pDeviceResources->GetBackBufferRenderTarget());

	Scenes.emplace_back(new Scene);
	auto& selector = Scenes.back();
	selector->SetRenderDeviceAndContext(pDevice.Get(), pContext.Get());
	selector->SetHudRenderDevice(pDeviceResources->GetD2DFactory(), pDeviceResources->GetD2DDeviceContext(), pDeviceResources->GetDWriteFactory());
	selector->SetCanvas(pDeviceResources->GetBackBufferRenderTarget());

	concurrency::task<void> loadScene([sceneFile,&selector]() {
		cout << "Current Directory :" << sys::current_path() << endl;
		cout << "Loading [Scene](" << sceneFile << ") ..." << endl;
		CoInitializeEx(NULL, COINIT::COINIT_APARTMENTTHREADED);
		selector->LoadFromFile(sceneFile.string());
		CoUninitialize();
		cout << "[Scene] Loading Finished!";
	});

	return true;
}

void App::SetupDevices(const ParamArchive* arch)
{
	bool enable = false;
	auto setting = arch->FirstChildElement("vicon");
	GetParam(setting, "enable", enable);
	if (setting && enable)
	{
		pVicon = Devices::IViconClient::Create();
		if (pVicon)
		{
			pVicon->Initialize(setting);
			pVicon->Start();
		}
	}

	setting = arch->FirstChildElement("leap");
	GetParam(setting, "enable", enable);
	if (setting && enable)
	{
		pLeap = Devices::LeapMotion::GetForCurrentView();
	}

	setting = arch->FirstChildElement("kinect");
	GetParam(setting, "enable", enable);
	if (setting && enable)
	{
		pKinect = Devices::KinectSensor::GetForCurrentView();
		if (pKinect)
		{
			XMMATRIX kinectCoord = XMMatrixRigidTransform(
				XMQuaternionRotationRollPitchYaw(-XM_PI / 12.0f, XM_PI, 0), // Orientation
				XMVectorSet(0, 0.0, 1.0f, 1.0f)); // Position
			pKinect->SetDeviceCoordinate(kinectCoord);
		}
	}

}

void App::RegisterComponent(IAppComponent *pComponent)
{
	auto pCursorInteractive = pComponent->As<ICursorInteractive>();
	auto& Regs = ComponentsEventRegisterations[pComponent];
	if (pCursorInteractive)
	{
		Regs.push_back(pWindow->CursorButtonDown += MakeEventHandler(&ICursorInteractive::OnMouseButtonDown, pCursorInteractive));
		Regs.push_back(pWindow->CursorButtonUp += MakeEventHandler(&ICursorInteractive::OnMouseButtonUp, pCursorInteractive));
		Regs.push_back(pWindow->CursorMove += MakeEventHandler(&ICursorInteractive::OnMouseMove, pCursorInteractive));
	}
	auto pKeyInteractive = pComponent->As<IKeybordInteractive>();
	if (pKeyInteractive)
	{
		Regs.push_back(pWindow->KeyDown += MakeEventHandler(&IKeybordInteractive::OnKeyDown, pKeyInteractive));
		Regs.push_back(pWindow->KeyUp += MakeEventHandler(&IKeybordInteractive::OnKeyUp, pKeyInteractive));
	}
	//auto pAnimatable = pComponent->As<ITimeAnimatable>();
	//if (pAnimatable)
	//	Regs.push_back(TimeElapsed += MakeEventHandler(&ITimeAnimatable::UpdateAnimation, pAnimatable));
	auto pHands = pComponent->As<IUserHandsInteractive>();
	if (pHands && pLeap)
	{
		Regs.push_back(pLeap->HandsTracked += MakeEventHandler(&IUserHandsInteractive::OnHandsTracked, pHands));
		Regs.push_back(pLeap->HandsLost += MakeEventHandler(&IUserHandsInteractive::OnHandsTrackLost, pHands));
		Regs.push_back(pLeap->HandsMove += MakeEventHandler(&IUserHandsInteractive::OnHandsMove, pHands));
	}
	//Components.push_back(std::move(pComponent));
}

void App::UnregisterComponent(IAppComponent * pComponent)
{
	auto itr = ComponentsEventRegisterations.find(pComponent);
	if (itr != ComponentsEventRegisterations.end())
		for (auto& connection : itr->second)
			connection.disconnect();
}


void App::OnExit()
{
}

bool App::OnIdle()
{
	if (pLeap)
		pLeap->PullFrame();

	if (pVicon)
		pVicon->Update();

	//if (pKinect)
	//	pKinect->ProcessFrame();

	for (auto& pScene : Scenes)
	{
		pScene->Update();
	}

	pDeviceResources->GetBackBufferRenderTarget().Clear(pContext.Get());
	for (auto& pScene : Scenes)
	{
		pScene->Render(pContext.Get());
	}

	pDeviceResources->Present();

	return true;
}

void App::OnDeviceLost()
{
}

void App::OnDeviceRestored()
{
}

void App::OnResize(const Vector2 & size)
{
	//pDeviceResources->SetLogicalSize(DeviceResources::Size(size.x,size.y));
	//auto& bb = pDeviceResources->GetBackBufferRenderTarget();
	//for (auto& scene : Scenes)
	//{
	//	scene->SetCanvas(bb);
	//}
}

path App::GetResourcesDirectory() const
{
	return m_assetsDir.wstring();
}

void App::SetResourcesDirectory(const std::wstring & dir)
{
	m_assetsDir = dir;
}

void XM_CALLCONV App::RenderToView(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection)
{
	//auto pContext = pDeviceResources->GetD3DDeviceContext();
	//for (auto& pScene : Components)
	//{
	//	auto pViewable = dynamic_cast<IViewable*>(pScene.get());
	//	if (pViewable)
	//	{
	//		pViewable->UpdateViewMatrix(view,projection);
	//		//pViewable->UpdateProjectionMatrix(projection);
	//	}
	//	//auto pRenderable = pScene->As<IRenderable>();
	//	//if (pRenderable)
	//	//	pRenderable->Render(pContext);
	//}
}

//inline void SampleListener::onConnect(const Leap::Controller & controller) {
//	std::cout << "Connected" << std::endl;
//}
//
//inline void SampleListener::onFrame(const Leap::Controller & controller) {
//	auto frame = controller.frame();
//	if (frame.hands().count() > PrevHandsCount)
//	{
//		PrevHandsCount = frame.hands().count();
//	}
//	else (frame.hands().count() < PrevHandsCount)
//	{
//		PrevHandsCount = frame.hands().count();
//	}
//	//std::cout << "Frame available" << std::endl;
//}

