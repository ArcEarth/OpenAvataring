#include "pch_bcl.h"
#include "OculusRift.h"
#define OVR_D3D_VERSION 11
#include <OVR_CAPI.h>
#include <OVR_CAPI_D3D.h> // God dame this to put a important header inside here!
#include <SpriteBatch.h>

#pragma comment(lib,"LibOVR.lib")

using namespace Causality;
using namespace Causality::Devices;
using namespace std;

class OculusRift::Impl
{
public:
	static int InstanceCount;
	int				   HMDIndex;
	ID3D11DeviceContext			   *pDeviceContext;
	DirectX::RenderTarget			EyeTextures[2];
	//DirectX::DepthStencilBuffer	   DepthStencilBuffer;

	ovrHmd             HMD;
	ovrGraphicsLuid	   luid;
	ovrTrackingState   OvrTrackingState;
	ovrD3D11Texture	   OvrEyeTextures[2];
	ovrPosef		   OvrEyePose[2];
	ovrEyeRenderDesc   EyeRenderDesc[2];
	ovrFrameTiming	   OvrFrameTiming;
	ovrVector3f		   HmdToEyeViewOffset[2];

	Impl(int index = 0)
		: HMDIndex(index)
	{
		if (InstanceCount == 0)
		{
			if (!ovr_Initialize())
				return;
		}

		ovrResult result = ovr_Create(&HMD, &luid);
		if (!OVR_SUCCESS(result))
			throw runtime_error("Oculus Rift creation failed.");

		if (HMD)
			InstanceCount++;
	}

	bool IsOk() const { return (bool)HMD; }

	~Impl()
	{
		if (HMD)
			ovr_Destroy(HMD);
		InstanceCount--;
		if (InstanceCount == 0)
		{
			ovr_Shutdown();
		}
	}

	HRESULT Initialize(HWND hWnd, DirectX::DeviceResources* pDeviceResource)
	{
		ovrBool Result;
		pDeviceContext = pDeviceResource->GetD3DDeviceContext();
		if (!HMD)
		{
			//MessageBoxA(NULL, "Oculus Rift not detected.", "", MB_OK);
			return(E_FAIL);
		}
		//Setup Window and Graphics - use window frame if relying on Oculus driver
		const int backBufferMultisample = pDeviceResource->GetMultiSampleCount();

		auto desc = ovr_GetHmdDesc(HMD);
		
		//Configure Stereo settings.
		auto recommenedTex0Size = ovr_GetFovTextureSize(HMD, ovrEye_Left, desc.DefaultEyeFov[0], 1.0f);
		auto recommenedTex1Size = ovr_GetFovTextureSize(HMD, ovrEye_Right, desc.DefaultEyeFov[1], 1.0f);
		ovrSizei RenderTargetSize;
		RenderTargetSize.w = recommenedTex0Size.w + recommenedTex1Size.w;
		RenderTargetSize.h = max(recommenedTex0Size.h, recommenedTex1Size.h);

		const int eyeRenderMultisample = 1;
		{
			DirectX::RenderTarget target(pDeviceResource->GetD3DDevice(), RenderTargetSize.w, RenderTargetSize.h);
			//DepthStencilBuffer = DirectX::DepthStencilBuffer(pDeviceResource->GetD3DDevice(), RenderTargetSize.w, RenderTargetSize.h);
			auto w = target.ViewPort().Width;
			auto h = target.ViewPort().Height;
			D3D11_VIEWPORT viewport;
			ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
			viewport.TopLeftX = 0;
			viewport.TopLeftY = 0;
			viewport.Width = (float)w / 2;
			viewport.Height = (float)h;

			EyeTextures[0] = target.Subview(viewport);
			viewport.TopLeftX = (float)w / 2;
			EyeTextures[1] = target.Subview(viewport);

			// Query D3D texture data.
			OvrEyeTextures[0].D3D11.Header.API = ovrRenderAPI_D3D11;
			OvrEyeTextures[0].D3D11.Header.TextureSize = RenderTargetSize;
			OvrEyeTextures[0].D3D11.Header.RenderViewport = OVR::Recti(0, 0, w / 2, h);
			OvrEyeTextures[0].D3D11.pTexture = EyeTextures[0].ColorBuffer();
			OvrEyeTextures[0].D3D11.pSRView = EyeTextures[0].ColorBuffer();

			// Right eye uses the same texture, but different rendering viewport.
			OvrEyeTextures[1] = OvrEyeTextures[0];
			OvrEyeTextures[1].D3D11.Header.RenderViewport = OVR::Recti(w / 2, 0, w / 2, h);
		}


		// Initialize eye rendering information.
		// The viewport sizes are re-computed in case RenderTargetSize changed due to HW limitations.
		ovrFovPort eyeFov[2] = { desc.DefaultEyeFov[0],desc.DefaultEyeFov[1] };

		auto outputsize = pDeviceResource->GetOutputSize();

		// Configure d3d11.
		ovrD3D11Config d3d11cfg;
		d3d11cfg.D3D11.Header.API = ovrRenderAPI_D3D11;
		d3d11cfg.D3D11.Header.BackBufferSize = ovrSizei{ (int)outputsize.Width, (int)outputsize.Height };
		d3d11cfg.D3D11.Header.Multisample = backBufferMultisample;
		d3d11cfg.D3D11.pDevice = pDeviceResource->GetD3DDevice();
		d3d11cfg.D3D11.pDeviceContext = pDeviceResource->GetD3DDeviceContext();
		d3d11cfg.D3D11.pBackBufferRT = pDeviceResource->GetBackBufferRenderTargetView();
		d3d11cfg.D3D11.pSwapChain = pDeviceResource->GetSwapChain();

		OVR_D3D_VERSION
		Result = ovrHmd_ConfigureRendering(HMD, &d3d11cfg.Config,
			ovrDistortionCap_Chromatic | ovrDistortionCap_HqDistortion,
			eyeFov, EyeRenderDesc);

		HmdToEyeViewOffset[0] = EyeRenderDesc[0].HmdToEyeViewOffset;
		HmdToEyeViewOffset[1] = EyeRenderDesc[1].HmdToEyeViewOffset;

		if (!Result)
		{
			//MessageBoxA(NULL, "Oculus Rift Rendering failed to config.", "", MB_OK);
			return(E_FAIL);
		}

		Result = ovrHmd_AttachToWindow(HMD, hWnd, NULL, NULL);
		if (!Result)
			return (E_FAIL);

		ovrHmd_SetEnabledCaps(HMD, ovrHmdCap_LowPersistence | ovrHmdCap_DynamicPrediction);

		// Start the sensor which informs of the Rift's pose and motion
		Result = ovrHmd_ConfigureTracking(HMD, ovrTrackingCap_Orientation |
			ovrTrackingCap_MagYawCorrection |
			ovrTrackingCap_Position, 0);

		if (!Result)
		{
			//MessageBoxA(NULL, "Oculus Rift Tracking failed to config.", "", MB_OK);
			return(E_FAIL);
		}

		return S_OK;
	}

};

int OculusRift::Impl::InstanceCount = 0;

DirectX::XMMATRIX OculusRift::EyeProjection(EyesEnum eye) const
{
	const auto & fov = pImpl->HMD->DefaultEyeFov[eye];
	float l = 0.01f;
	return DirectX::XMMatrixPerspectiveOffCenterRH(-fov.LeftTan * l, fov.RightTan * l, -fov.DownTan * l, fov.UpTan * l, l, 100.0f);
}


OculusRift::OculusRift()
{
}

OculusRift::~OculusRift()
{

}

std::weak_ptr<OculusRift> OculusRift::wpCurrentDevice;

std::shared_ptr<OculusRift> Causality::Devices::OculusRift::GetForCurrentView()
{
	if (wpCurrentDevice.expired())
	{
		auto pDevice = Create();
		wpCurrentDevice = pDevice;
		return pDevice;
	}
	else
	{
		return wpCurrentDevice.lock();
	}
}

std::shared_ptr<OculusRift> OculusRift::Create(int hmdIdx)
{
	auto pRift = std::make_shared<OculusRift>();
	pRift->pImpl = std::make_unique<OculusRift::Impl>(hmdIdx);
	if (!pRift->pImpl->IsOk())
		return nullptr;
	else 
		return pRift;
}

bool OculusRift::Initialize(void)
{
	if (!ovr_InitializeRenderingShim())
		return false;
	return (bool)ovr_Initialize();
}

void OculusRift::DissmisHealthWarnning()
{
	if (pImpl->HMD)
		ovrHmd_DismissHSWDisplay(pImpl->HMD);
}

void OculusRift::BeginFrame()
{
	pImpl->EyeTextures[0].Clear(pImpl->pDeviceContext, DirectX::Colors::Green);
	pImpl->EyeTextures[0].Clear(pImpl->pDeviceContext);
	pImpl->OvrFrameTiming = ovrHmd_BeginFrame(pImpl->HMD, 0);
}
void OculusRift::EndFrame()
{

	ovrHmd_EndFrame(pImpl->HMD, pImpl->OvrEyePose, &pImpl->OvrEyeTextures->Texture);

}
bool OculusRift::InitializeGraphics(HWND hWnd, DirectX::DeviceResources* pDeviceResource)
{
	HRESULT hr = pImpl->Initialize(hWnd, pDeviceResource);
	if (FAILED(hr))
	{
		pImpl = nullptr;
		return false;
	}
	return true;
}

DirectX::Vector2 OculusRift::Resoulution() const
{
	return Vector2(pImpl->HMD->Resolution.w, pImpl->HMD->Resolution.h);
}

DirectX::Vector2 OculusRift::DesktopWindowPosition() const
{
	return Vector2(pImpl->HMD->WindowsPos.x, pImpl->HMD->WindowsPos.y);
}

const char * OculusRift::DisplayDeviceName() const
{
	return pImpl->HMD->DisplayDeviceName;
}

DirectX::RenderTarget& OculusRift::ViewTarget(EyesEnum eye)
{
	return pImpl->EyeTextures[(size_t) eye];
}

DirectX::RenderableTexture2D& OculusRift::ColorBuffer()
{
	return pImpl->EyeTextures[0].ColorBuffer();
}

DirectX::DepthStencilBuffer& OculusRift::DepthStencilBuffer()
{
	return pImpl->EyeTextures[0].DepthBuffer();
}

const StaticPose& OculusRift::EyePoses(EyesEnum eye) const
{
	auto& pose = pImpl->OvrEyePose[eye];
	ovrHmd_GetEyePoses(pImpl->HMD,0, pImpl->HmdToEyeViewOffset,&pImpl->OvrEyePose[eye], &pImpl->OvrTrackingState);
	return reinterpret_cast<StaticPose&>(pose);
}

const DynamicPose& OculusRift::HeadPose() const
{
	pImpl->OvrTrackingState = ovrHmd_GetTrackingState(pImpl->HMD, pImpl->OvrFrameTiming.ScanoutMidpointSeconds);
	return reinterpret_cast<DynamicPose&>(pImpl->OvrTrackingState.HeadPose);
}

float OculusRift::UserEyeHeight() const
{
	return ovrHmd_GetFloat(pImpl->HMD, OVR_KEY_EYE_HEIGHT, 1.76f);
}
