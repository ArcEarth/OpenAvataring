#include "pch_directX.h"
#include "DeviceResources.h"
#include "DirectXHelper.h"
#include <algorithm>
#include <cmath>

#if defined (__cplusplus_winrt)
#include <windows.ui.xaml.media.dxinterop.h>
#endif

using namespace D2D1;
using namespace DirectX;
using namespace Microsoft::WRL;
#if defined (__cplusplus_winrt)
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml::Controls;
using namespace Platform;
#endif

// Constants used to calculate screen rotations.
namespace ScreenRotation
{
	// 0-degree Z-rotation
	static const XMFLOAT4X4 Rotation0( 
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
		);

	// 90-degree Z-rotation
	static const XMFLOAT4X4 Rotation90(
		0.0f, 1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
		);

	// 180-degree Z-rotation
	static const XMFLOAT4X4 Rotation180(
		-1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, -1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
		);

	// 270-degree Z-rotation
	static const XMFLOAT4X4 Rotation270( 
		0.0f, -1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
		);
};

// Constructor for DeviceResources.
DirectX::DeviceResources::DeviceResources() : 
	m_screenViewport(),
	m_d3dFeatureLevel(D3D_FEATURE_LEVEL_9_1),
	m_d3dRenderTargetSize(),
	m_outputSize(),
	m_logicalSize(),
	m_nativeOrientation(DisplayOrientations::None),
	m_currentOrientation(DisplayOrientations::None),
	m_dpi(-1.0f),
	m_dpiScaleX(1.0f),
	m_dpiScaleY(1.0f),
	m_deviceNotify(nullptr)
{
	m_multiSampleLevel = 1;
	m_multiSampleQuality = 0;
	CreateDeviceIndependentResources();
	CreateDeviceResources();
}

void DirectX::DeviceResources::SetNativeWindow(HWND hWnd)
{
	m_deviceHostType = DeviceHostType::NativeWindow;
	RECT rect;
	GetWindowRect(hWnd, &rect);
	m_hWnd = hWnd;
	m_logicalSize = Size((float)(rect.right - rect.left),(float)(rect.bottom - rect.top));
	m_nativeOrientation = DisplayOrientations::None;
	m_currentOrientation = DisplayOrientations::None;

	UINT arrayElements = 4, modeElements = 4;
	DISPLAYCONFIG_PATH_INFO dispInfos[4];
	DISPLAYCONFIG_MODE_INFO dispModeInfo[4];
	QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &arrayElements, dispInfos, &modeElements, dispModeInfo,NULL);
	HDC screen = GetDC(hWnd);
	m_dpi = (float)GetDeviceCaps(screen, LOGPIXELSX);
	auto pyhsicalw = (float) GetDeviceCaps(screen, HORZRES);
	auto physicalh = (float) GetDeviceCaps(screen, VERTRES);
	m_dpiScaleY = (float) dispModeInfo[1].sourceMode.height / physicalh;
	m_dpiScaleX = (float) dispModeInfo[1].sourceMode.width / pyhsicalw;
	ReleaseDC(hWnd, screen);

	//m_d2dContext->SetDpi(m_dpi, m_dpi);

	CreateWindowSizeDependentResources();
}

#if defined (__cplusplus_winrt)
// This method is called when the CoreWindow is created (or re-created).
void DirectX::DeviceResources::SetCoreWindow(CoreWindow^ window)
{
	m_deviceHostType = DeviceHostType::CoreWindow;
	DisplayInformation^ currentDisplayInformation = DisplayInformation::GetForCurrentView();

	m_window = window;
	m_logicalSize = Size(window->Bounds.Width, window->Bounds.Height);
	m_nativeOrientation = currentDisplayInformation->NativeOrientation;
	m_currentOrientation = currentDisplayInformation->CurrentOrientation;
	m_dpi = currentDisplayInformation->LogicalDpi;
	m_d2dContext->SetDpi(m_dpi, m_dpi);

	CreateWindowSizeDependentResources();
}

// This method is called when the XAML control is created (or re-created).
void DirectX::DeviceResources::SetSwapChainPanel(SwapChainPanel^ panel)
{
	m_deviceHostType = DeviceHostType::Composition;
	DisplayInformation^ currentDisplayInformation = DisplayInformation::GetForCurrentView();

	m_swapChainPanel = panel;
	m_logicalSize = Size(static_cast<float>(panel->ActualWidth), static_cast<float>(panel->ActualHeight));
	m_nativeOrientation = currentDisplayInformation->NativeOrientation;
	m_currentOrientation = currentDisplayInformation->CurrentOrientation;
	m_dpiScaleX = panel->CompositionScaleX;
	m_dpiScaleY = panel->CompositionScaleY;
	m_dpi = currentDisplayInformation->LogicalDpi;
	m_d2dContext->SetDpi(m_dpi, m_dpi);

	CreateWindowSizeDependentResources();
}
#endif


// Configures resources that don't depend on the Direct3D device.
void DirectX::DeviceResources::CreateDeviceIndependentResources()
{
	// Initialize Direct2D resources.
	D2D1_FACTORY_OPTIONS options;
	ZeroMemory(&options, sizeof(D2D1_FACTORY_OPTIONS));

#if defined(_DEBUG)
	// If the project is in a debug build, enable Direct2D debugging via SDK Layers.
	options.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif

	// Initialize the Direct2D Factory.
	DirectX::ThrowIfFailed(
		D2D1CreateFactory(
			D2D1_FACTORY_TYPE_SINGLE_THREADED,
			__uuidof(ID2D1Factory2),
			&options,
			&m_d2dFactory
			)
		);

	// Initialize the DirectWrite Factory.
	DirectX::ThrowIfFailed(
		DWriteCreateFactory(
			DWRITE_FACTORY_TYPE_SHARED,
			__uuidof(IDWriteFactory2),
			&m_dwriteFactory
			)
		);

	// Initialize the Windows Imaging Component (WIC) Factory.
	//DirectX::ThrowIfFailed(
	//	CoCreateInstance(
	//		CLSID_WICImagingFactory2,
	//		nullptr,
	//		CLSCTX_INPROC_SERVER,
	//		IID_PPV_ARGS(&m_wicFactory)
	//		)
	//	);
}

// Configures the Direct3D device, and stores handles to it and the device context.
void DirectX::DeviceResources::CreateDeviceResources() 
{
	// This flag adds support for surfaces with a different color channel ordering
	// than the API default. It is required for compatibility with Direct2D.
	UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
	if (DirectX::SdkLayersAvailable())
	{
		// If the project is in a debug build, enable debugging via SDK Layers with this flag.
		creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
	}
#endif

	// This array defines the set of DirectX hardware feature levels this app will support.
	// Note the ordering should be preserved.
	// Don't forget to declare your application's minimum required feature level in its
	// description.  All applications are assumed to support 9.1 unless otherwise stated.
	D3D_FEATURE_LEVEL featureLevels[] = 
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		D3D_FEATURE_LEVEL_9_3,
		D3D_FEATURE_LEVEL_9_2,
		D3D_FEATURE_LEVEL_9_1
	};

	// Create the Direct3D 11 API device object and a corresponding context.
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> context;

	HRESULT hr = D3D11CreateDevice(
		nullptr,					// Specify nullptr to use the default adapter.
		D3D_DRIVER_TYPE_HARDWARE,	// Create a device using the hardware graphics driver.
		0,							// Should be 0 unless the driver is D3D_DRIVER_TYPE_SOFTWARE.
		creationFlags,				// Set debug and Direct2D compatibility flags.
		featureLevels,				// List of feature levels this app can support.
		ARRAYSIZE(featureLevels),	// Size of the list above.
		D3D11_SDK_VERSION,			// Always set this to D3D11_SDK_VERSION for Windows Store apps.
		&device,					// Returns the Direct3D device created.
		&m_d3dFeatureLevel,			// Returns feature level of device created.
		&context					// Returns the device immediate context.
		);

	if (FAILED(hr))
	{
		// If the initialization fails, fall back to the WARP device.
		// For more information on WARP, see: 
		// http://go.microsoft.com/fwlink/?LinkId=286690
		DirectX::ThrowIfFailed(
			D3D11CreateDevice(
				nullptr,
				D3D_DRIVER_TYPE_WARP, // Create a WARP device instead of a hardware device.
				0,
				creationFlags,
				featureLevels,
				ARRAYSIZE(featureLevels),
				D3D11_SDK_VERSION,
				&device,
				&m_d3dFeatureLevel,
				&context
				)
			);
	}

	// Store pointers to the Direct3D 11.1 API device and immediate context.
	DirectX::ThrowIfFailed(
		device.As(&m_d3dDevice)
		);

	DirectX::ThrowIfFailed(
		context.As(&m_d3dContext)
		);

	// Create the Direct2D device object and a corresponding context.
	ComPtr<IDXGIDevice3> dxgiDevice;
	DirectX::ThrowIfFailed(
		m_d3dDevice.As(&dxgiDevice)
		);

	DirectX::ThrowIfFailed(
		m_d2dFactory->CreateDevice(dxgiDevice.Get(), &m_d2dDevice)
		);

	DirectX::ThrowIfFailed(
		m_d2dDevice->CreateDeviceContext(
			D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
			&m_d2dContext
			)
		);

}

void DirectX::DeviceResources::CreateSwapChainForComposition(IDXGIFactory2* dxgiFactory, DXGI_SWAP_CHAIN_DESC1 *pSwapChainDesc)
{
#if defined (__cplusplus_winrt)
	// When using XAML interop, the swap chain must be created for composition.
	DirectX::ThrowIfFailed(
		dxgiFactory->CreateSwapChainForComposition(
			m_d3dDevice.Get(),
			pSwapChainDesc,
			nullptr,
			&m_swapChain
			)
		);	// Associate swap chain with SwapChainPanel
	// UI changes will need to be dispatched back to the UI thread
	m_swapChainPanel->Dispatcher->RunAsync(CoreDispatcherPriority::High, ref new DispatchedHandler([=]()
	{
		// Get backing native interface for SwapChainPanel
		ComPtr<ISwapChainPanelNative> panelNative;
		DirectX::ThrowIfFailed(
			reinterpret_cast<IUnknown*>(m_swapChainPanel)->QueryInterface(IID_PPV_ARGS(&panelNative))
			);

		DirectX::ThrowIfFailed(
			panelNative->SetSwapChain(m_swapChain.Get())
			);
	}, CallbackContext::Any));
#else

#endif
}


// These resources need to be recreated every time the window size is changed.
void DirectX::DeviceResources::CreateWindowSizeDependentResources() 
{
	auto SwapChainFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
	// Clear the previous window size specific context.
	ID3D11RenderTargetView* nullViews[] = {nullptr};
	m_d3dContext->OMSetRenderTargets(ARRAYSIZE(nullViews), nullViews, nullptr);
	m_d3dRenderTargetView = nullptr;
	m_d2dContext->SetTarget(nullptr);
	m_d2dTargetBitmap = nullptr;
	m_d3dDepthStencilView = nullptr;
	m_d3dContext->Flush();

	// Calculate the necessary swap chain and render target size in pixels.
	m_outputSize.Width = m_logicalSize.Width * m_dpiScaleX;
	m_outputSize.Height = m_logicalSize.Height * m_dpiScaleY;
	
	// Prevent zero size DirectX content from being created.
	m_outputSize.Width = std::max(m_outputSize.Width, 1.0f);
	m_outputSize.Height = std::max(m_outputSize.Height, 1.0f);

	// The width and height of the swap chain must be based on the window's
	// natively-oriented width and height. If the window is not in the native
	// orientation, the dimensions must be reversed.
	DXGI_MODE_ROTATION displayRotation = ComputeDisplayRotation();

	bool swapDimensions = displayRotation == DXGI_MODE_ROTATION_ROTATE90 || displayRotation == DXGI_MODE_ROTATION_ROTATE270;
	m_d3dRenderTargetSize.Width = swapDimensions ? m_outputSize.Height : m_outputSize.Width;
	m_d3dRenderTargetSize.Height = swapDimensions ? m_outputSize.Width : m_outputSize.Height;

	if (m_swapChain != nullptr)
	{
		// If the swap chain already exists, resize it.
		HRESULT hr = m_swapChain->ResizeBuffers(
			2, // Double-buffered swap chain.
			lround(m_d3dRenderTargetSize.Width),
			lround(m_d3dRenderTargetSize.Height),
			SwapChainFormat,
			0
			);

		if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
		{
			// If the device was removed for any reason, a new device and swap chain will need to be created.
			HandleDeviceLost();

			// Everything is set up now. Do not continue execution of this method. HandleDeviceLost will reenter this method 
			// and correctly set up the new device.
			return;
		}
		else
		{
			DirectX::ThrowIfFailed(hr);
		}
	}
	else
	{
		// Otherwise, create a new one using the same adapter as the existing Direct3D device.
		// This sequence obtains the DXGI factory that was used to create the Direct3D device above.
		ComPtr<IDXGIDevice3> dxgiDevice;
		DirectX::ThrowIfFailed(
			m_d3dDevice.As(&dxgiDevice)
			);

		ComPtr<IDXGIAdapter> dxgiAdapter;
		DirectX::ThrowIfFailed(
			dxgiDevice->GetAdapter(&dxgiAdapter)
			);

		ComPtr<IDXGIFactory2> dxgiFactory;
		DirectX::ThrowIfFailed(
			dxgiAdapter->GetParent(IID_PPV_ARGS(&dxgiFactory))
			);

		DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {0};

		if (m_deviceHostType != DirectX::DeviceHostType::NativeWindow)
		{
			m_multiSampleLevel = 1;
			m_multiSampleQuality = 0;
		}

		if (m_multiSampleLevel > 1)
		{
			HRESULT hr = m_d3dDevice->CheckMultisampleQualityLevels(SwapChainFormat, m_multiSampleLevel, &m_multiSampleQuality);
			m_multiSampleQuality = std::min(std::max(m_multiSampleQuality - 1U, 0U),4U);
			swapChainDesc.SampleDesc.Count = m_multiSampleLevel;
			swapChainDesc.SampleDesc.Quality = m_multiSampleQuality;
			swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
		}
		else
		{
			swapChainDesc.SampleDesc.Count = 1; // Don't use multi-sampling.
			swapChainDesc.SampleDesc.Quality = 0;
			swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL; // All Windows Store apps must use this SwapEffect.
		}

		swapChainDesc.Width = lround(m_d3dRenderTargetSize.Width); // Match the size of the window.
		swapChainDesc.Height = lround(m_d3dRenderTargetSize.Height);
		swapChainDesc.Format = SwapChainFormat; // This is the most common swap chain format.
		swapChainDesc.Stereo = false;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.BufferCount = 2; // Use double-buffering to minimize latency.
		swapChainDesc.Flags = 0;
		swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
		swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

		switch (m_deviceHostType)
		{
		case DirectX::DeviceHostType::None:
			break;
		case DirectX::DeviceHostType::NativeWindow:
			DirectX::ThrowIfFailed(
				dxgiFactory->CreateSwapChainForHwnd(
				m_d3dDevice.Get(),
				m_hWnd,
				&swapChainDesc,
				nullptr,
				nullptr,
				&m_swapChain)
				);
			break;
#if defined (__cplusplus_winrt)
		case DirectX::DeviceHostType::CoreWindow:
			DirectX::ThrowIfFailed(
				dxgiFactory->CreateSwapChainForCoreWindow(
				m_d3dDevice.Get(),
				reinterpret_cast<IUnknown*>(m_window.Get()),
				&swapChainDesc,
				nullptr,
				&m_swapChain
				)
				);
			break;
#endif
		case DirectX::DeviceHostType::Composition:
			CreateSwapChainForComposition(dxgiFactory.Get(), &swapChainDesc);
			break;
		default:
			break;
		}
		
		// Ensure that DXGI does not queue more than one frame at a time. This both reduces latency and
		// ensures that the application will only render after each VSync, minimizing power consumption.
		DirectX::ThrowIfFailed(
			dxgiDevice->SetMaximumFrameLatency(1)
			);
	}

	// Set the proper orientation for the swap chain, and generate 2D and
	// 3D matrix transformations for rendering to the rotated swap chain.
	// Note the rotation angle for the 2D and 3D transforms are different.
	// This is due to the difference in coordinate spaces.  Additionally,
	// the 3D matrix is specified explicitly to avoid rounding errors.

	switch (displayRotation)
	{
	case DXGI_MODE_ROTATION_UNSPECIFIED:
	case DXGI_MODE_ROTATION_IDENTITY:
		m_orientationTransform2D = Matrix3x2F::Identity();
		m_orientationTransform3D = ScreenRotation::Rotation0;
		break;

	case DXGI_MODE_ROTATION_ROTATE90:
		m_orientationTransform2D = 
			Matrix3x2F::Rotation(90.0f) *
			Matrix3x2F::Translation(m_logicalSize.Height, 0.0f);
		m_orientationTransform3D = ScreenRotation::Rotation270;
		break;

	case DXGI_MODE_ROTATION_ROTATE180:
		m_orientationTransform2D = 
			Matrix3x2F::Rotation(180.0f) *
			Matrix3x2F::Translation(m_logicalSize.Width, m_logicalSize.Height);
		m_orientationTransform3D = ScreenRotation::Rotation180;
		break;

	case DXGI_MODE_ROTATION_ROTATE270:
		m_orientationTransform2D = 
			Matrix3x2F::Rotation(270.0f) *
			Matrix3x2F::Translation(0.0f, m_logicalSize.Width);
		m_orientationTransform3D = ScreenRotation::Rotation90;
		break;

	default:
		throw std::exception("Failed to initialize");
	}

	// Roatation is only avaiable for them
	if (m_multiSampleLevel <= 1)
	{
		HRESULT hr =
			m_swapChain->SetRotation(displayRotation);
		if (FAILED(hr) && hr != DXGI_ERROR_INVALID_CALL)
		{
			ThrowIfFailed(hr);
		}
	}

	if (m_deviceHostType == DeviceHostType::Composition)
	{
		// Setup inverse scale on the swap chain
		DXGI_MATRIX_3X2_F inverseScale = { 0 };
		inverseScale._11 = 1.0f / m_dpiScaleX;
		inverseScale._22 = 1.0f / m_dpiScaleY;
		ComPtr<IDXGISwapChain2> spSwapChain2;
		DirectX::ThrowIfFailed(
			m_swapChain.As<IDXGISwapChain2>(&spSwapChain2)
			);

		DirectX::ThrowIfFailed(
			spSwapChain2->SetMatrixTransform(&inverseScale)
			);
	}

	// Create a render target view of the swap chain back buffer.
	ComPtr<ID3D11Texture2D> backBuffer;
	DirectX::ThrowIfFailed(
		m_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer))
		);

	DirectX::ThrowIfFailed(
		m_d3dDevice->CreateRenderTargetView(
			backBuffer.Get(),
			nullptr,
			&m_d3dRenderTargetView
			)
		);

	m_ColorBackBuffer = RenderableTexture2D(backBuffer.Get(), m_d3dRenderTargetView.Get(), nullptr);

	// Create a depth stencil view for use with 3D rendering if needed.
	CD3D11_TEXTURE2D_DESC depthStencilDesc(
		DXGI_FORMAT_D24_UNORM_S8_UINT, 
		lround(m_d3dRenderTargetSize.Width),
		lround(m_d3dRenderTargetSize.Height),
		1, // This depth stencil view has only one texture.
		1, // Use a single mipmap level.
		D3D11_BIND_DEPTH_STENCIL,
		D3D11_USAGE_DEFAULT,
		0U,
		m_multiSampleLevel,
		m_multiSampleQuality
		);

	ComPtr<ID3D11Texture2D> depthStencil;
	DirectX::ThrowIfFailed(
		m_d3dDevice->CreateTexture2D(
			&depthStencilDesc,
			nullptr,
			&depthStencil
			)
		);

	//CD3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc(depthStencil.Get(),D3D11_DSV_DIMENSION_TEXTURE2D);

	DirectX::ThrowIfFailed(
		m_d3dDevice->CreateDepthStencilView(
			depthStencil.Get(),
			nullptr,
			&m_d3dDepthStencilView
			)
		);
	
	m_DepthBuffer = DepthStencilBuffer(depthStencil.Get(),m_d3dDepthStencilView.Get());

	// Set the 3D rendering viewport to target the entire window.
	m_screenViewport = CD3D11_VIEWPORT(
		0.0f,
		0.0f,
		m_d3dRenderTargetSize.Width,
		m_d3dRenderTargetSize.Height
		);

	m_BackBuffer = RenderTarget(m_ColorBackBuffer, m_DepthBuffer, m_screenViewport);

	m_d3dContext->RSSetViewports(1, &m_screenViewport);

	// Create a Direct2D target bitmap associated with the
	// swap chain back buffer and set it as the current target.
	D2D1_BITMAP_PROPERTIES1 bitmapProperties = 
		D2D1::BitmapProperties1(
			D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
			D2D1::PixelFormat(SwapChainFormat, D2D1_ALPHA_MODE_PREMULTIPLIED),
			m_dpi,
			m_dpi
			);

	//? Critical!!!!
	//return;

	ComPtr<IDXGISurface2> dxgiBackBuffer;
	DirectX::ThrowIfFailed(
		m_swapChain->GetBuffer(0, IID_PPV_ARGS(&dxgiBackBuffer))
		);

	DirectX::ThrowIfFailed(
		m_d2dContext->CreateBitmapFromDxgiSurface(
			dxgiBackBuffer.Get(),
			&bitmapProperties,
			&m_d2dTargetBitmap
			)
		);

	m_d2dContext->SetTarget(m_d2dTargetBitmap.Get());

	// Grayscale text anti-aliasing is recommended for all Windows Store apps.
	m_d2dContext->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
}

// This method is called in the event handler for the SizeChanged event.
void DirectX::DeviceResources::SetLogicalSize(Size logicalSize)
{
	if (m_logicalSize != logicalSize)
	{
		m_logicalSize = logicalSize;
		CreateWindowSizeDependentResources();
	}
}

// This method is called in the event handler for the DpiChanged event.
void DirectX::DeviceResources::SetDpi(float dpi)
{
	if (dpi != m_dpi)
	{
		m_dpi = dpi;
		m_d2dContext->SetDpi(m_dpi, m_dpi);
		CreateWindowSizeDependentResources();
	}
}

// This method is called in the event handler for the OrientationChanged event.
void DirectX::DeviceResources::SetCurrentOrientation(DisplayOrientations currentOrientation)
{
	if (m_currentOrientation != currentOrientation)
	{
		m_currentOrientation = currentOrientation;
		CreateWindowSizeDependentResources();
	}
}

void DirectX::DeviceResources::SetCurrentOrientation(DirectX::FXMVECTOR quaternion)
{
	XMStoreFloat4x4(&m_orientationTransform3D, DirectX::XMMatrixRotationQuaternion(quaternion));
}


// This method is called in the event handler for the CompositionScaleChanged event.
void DirectX::DeviceResources::SetCompositionScale(float compositionScaleX, float compositionScaleY)
{
	if (m_dpiScaleX != compositionScaleX ||
		m_dpiScaleY != compositionScaleY)
	{
		m_dpiScaleX = compositionScaleX;
		m_dpiScaleY = compositionScaleY;
		CreateWindowSizeDependentResources();
	}
}

// This method is called in the event handler for the DisplayContentsInvalidated event.
void DirectX::DeviceResources::ValidateDevice()
{
	// The D3D Device is no longer valid if the default adapter changed since the device
    // was created or if the device has been removed.

	// First, get the information for the default adapter from when the device was created.

	ComPtr<IDXGIDevice3> dxgiDevice;
	DirectX::ThrowIfFailed(m_d3dDevice.As(&dxgiDevice));

	ComPtr<IDXGIAdapter> deviceAdapter;
	DirectX::ThrowIfFailed(dxgiDevice->GetAdapter(&deviceAdapter));

	ComPtr<IDXGIFactory2> deviceFactory;
	DirectX::ThrowIfFailed(deviceAdapter->GetParent(IID_PPV_ARGS(&deviceFactory)));

	ComPtr<IDXGIAdapter1> previousDefaultAdapter;
	DirectX::ThrowIfFailed(deviceFactory->EnumAdapters1(0, &previousDefaultAdapter));

	DXGI_ADAPTER_DESC previousDesc;
	DirectX::ThrowIfFailed(previousDefaultAdapter->GetDesc(&previousDesc));

	// Next, get the information for the current default adapter.

	ComPtr<IDXGIFactory2> currentFactory;
	DirectX::ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&currentFactory)));

	ComPtr<IDXGIAdapter1> currentDefaultAdapter;
	DirectX::ThrowIfFailed(currentFactory->EnumAdapters1(0, &currentDefaultAdapter));

	DXGI_ADAPTER_DESC currentDesc;
	DirectX::ThrowIfFailed(currentDefaultAdapter->GetDesc(&currentDesc));

	// If the adapter LUIDs don't match, or if the device reports that it has been removed,
	// a new D3D device must be created.

	if (previousDesc.AdapterLuid.LowPart != currentDesc.AdapterLuid.LowPart ||
		previousDesc.AdapterLuid.HighPart != currentDesc.AdapterLuid.HighPart ||
		FAILED(m_d3dDevice->GetDeviceRemovedReason()))
	{
		// Release references to resources related to the old device.
		dxgiDevice = nullptr;
		deviceAdapter = nullptr;
		deviceFactory = nullptr;
		previousDefaultAdapter = nullptr;

		// Create a new device and swap chain.
		HandleDeviceLost();
	}
}

// Recreate all device resources and set them back to the current state.
void DirectX::DeviceResources::HandleDeviceLost()
{
	m_swapChain = nullptr;

	if (m_deviceNotify != nullptr)
	{
		m_deviceNotify->OnDeviceLost();
	}

	CreateDeviceResources();
	m_d2dContext->SetDpi(m_dpi, m_dpi);
	CreateWindowSizeDependentResources();

	if (m_deviceNotify != nullptr)
	{
		m_deviceNotify->OnDeviceRestored();
	}
}

// Register our DeviceNotify to be informed on device lost and creation.
void DirectX::DeviceResources::RegisterDeviceNotify(DirectX::IDeviceNotify* deviceNotify)
{
	m_deviceNotify = deviceNotify;
}

// Call this method when the app suspends. It provides a hint to the driver that the app 
// is entering an idle state and that temporary buffers can be reclaimed for use by other apps.
void DirectX::DeviceResources::Trim()
{
	ComPtr<IDXGIDevice3> dxgiDevice;
	m_d3dDevice.As(&dxgiDevice);

	dxgiDevice->Trim();
}

// Present the contents of the swap chain to the screen.
void DirectX::DeviceResources::Present() 
{
	// The first argument instructs DXGI to block until VSync, putting the application
	// to sleep until the next VSync. This ensures we don't waste any cycles rendering
	// frames that will never be displayed to the screen.
	HRESULT hr = m_swapChain->Present(1, 0);

#if !defined(_DEBUG)
	// Discard the contents of the render target.
	// This is a valid operation only when the existing contents will be entirely
	// overwritten. If dirty or scroll rects are used, this call should be removed.
	m_d3dContext->DiscardView(m_d3dRenderTargetView.Get());

	// Discard the contents of the depth stencil.
	m_d3dContext->DiscardView(m_d3dDepthStencilView.Get());
#endif

	// If the device was removed either by a disconnection or a driver upgrade, we 
	// must recreate all device resources.
	if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
	{
		HandleDeviceLost();
	}
	else
	{
		DirectX::ThrowIfFailed(hr);
	}
}

// This method determines the rotation between the display device's native Orientation and the
// current display orientation.
DXGI_MODE_ROTATION DirectX::DeviceResources::ComputeDisplayRotation()
{

	if (m_deviceHostType == DeviceHostType::NativeWindow)
	{

		DEVMODE devmode;
		//DEVMODE structure 
		ZeroMemory(&devmode, sizeof(DEVMODE));

		devmode.dmSize = sizeof(DEVMODE);

		devmode.dmFields = DM_DISPLAYORIENTATION;
		//Check display orientation 
		EnumDisplaySettingsEx(NULL, ENUM_CURRENT_SETTINGS, &devmode, EDS_RAWMODE);
		
		switch (devmode.dmDisplayOrientation)
		{
		case DMDO_DEFAULT :
			return DXGI_MODE_ROTATION_IDENTITY;
		case DMDO_90:
			return DXGI_MODE_ROTATION_ROTATE90;
		case DMDO_180:
			return DXGI_MODE_ROTATION_ROTATE180;
		case DMDO_270:
			return DXGI_MODE_ROTATION_ROTATE270;
		default:
			return DXGI_MODE_ROTATION_UNSPECIFIED;
		} 
	}

	DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_UNSPECIFIED;

	// Note: NativeOrientation can only be Landscape or Portrait even though
	// the DisplayOrientations enum has other values.
	switch (m_nativeOrientation)
	{
	case DisplayOrientations::Landscape:
		switch (m_currentOrientation)
		{
		case DisplayOrientations::Landscape:
			rotation = DXGI_MODE_ROTATION_IDENTITY;
			break;

		case DisplayOrientations::Portrait:
			rotation = DXGI_MODE_ROTATION_ROTATE270;
			break;

		case DisplayOrientations::LandscapeFlipped:
			rotation = DXGI_MODE_ROTATION_ROTATE180;
			break;

		case DisplayOrientations::PortraitFlipped:
			rotation = DXGI_MODE_ROTATION_ROTATE90;
			break;
		}
		break;

	case DisplayOrientations::Portrait:
		switch (m_currentOrientation)
		{
		case DisplayOrientations::Landscape:
			rotation = DXGI_MODE_ROTATION_ROTATE90;
			break;

		case DisplayOrientations::Portrait:
			rotation = DXGI_MODE_ROTATION_IDENTITY;
			break;

		case DisplayOrientations::LandscapeFlipped:
			rotation = DXGI_MODE_ROTATION_ROTATE270;
			break;

		case DisplayOrientations::PortraitFlipped:
			rotation = DXGI_MODE_ROTATION_ROTATE180;
			break;
		}
		break;
	}
	return rotation;
}

GraphicsResource::~GraphicsResource() {
	//pDeviceResources->UnregisterDeviceNotify(this);
}

void GraphicsResource::SetDeviceResource(const std::shared_ptr<DeviceResources> pDeviceResources)
{
	if (m_pDeviceResources != pDeviceResources)
	{
		m_pDeviceResources = pDeviceResources;
		//pDeviceResources->RegisterDeviceNotify(this);
		this->CreateDeviceResources();
		this->CreateSizeDependentResource();
	}
}
