#pragma once
#define NOMINMAX
#if defined (__cplusplus_winrt)
#include <agile.h>
#endif
#include <wrl\client.h>
#include <Windows.h>
#include <d3d11_2.h>
#include <d2d1_2.h>
#include <dwrite_2.h>
#include <DirectXMath.h>
#include <string>
#include <mutex>
#include "Textures.h"


namespace DirectX
{
	class DeviceResources;
    // Provides an interface for an application that owns DeviceResources to be notified of the device being lost or created.
	interface IDeviceNotify
	{
		virtual void OnDeviceLost() = 0;
		virtual void OnDeviceRestored() = 0;
	};

	enum class DeviceHostType
	{
		None,
		NativeWindow,
		CoreWindow,
		Composition,
	};

	class IDeviceResouces abstract
	{
		virtual ID3D11Device2*			GetD3DDevice() const = 0;
		virtual ID3D11DeviceContext2*	GetD3DDeviceContext() const = 0;
		virtual ID2D1Factory2*			GetD2DFactory() const = 0;
		virtual ID2D1Device1*			GetD2DDevice() const = 0;
		virtual ID2D1DeviceContext1*	GetD2DDeviceContext() const = 0;
		virtual IDWriteFactory2*		GetDWriteFactory() const = 0;
	};

	// Controls all the DirectX device resources.
	class DeviceResources : public IDeviceResouces
	{
	public:
		DeviceResources();
		void SetNativeWindow(HWND hWnd);

#if defined (__cplusplus_winrt)
		typedef Windows::Foundation::Size Size;
		typedef Windows::Graphics::Display::DisplayOrientations DisplayOrientations;

		void SetCoreWindow(Windows::UI::Core::CoreWindow^ window);
		void SetSwapChainPanel(Windows::UI::Xaml::Controls::SwapChainPanel^ panel);
#else
		enum class DisplayOrientations : unsigned int
		{
			Landscape = 1,
			LandscapeFlipped = 4,
			None = 0,
			Portrait = 2,
			PortraitFlipped = 8,
		};

		struct Size
		{
			float Width, Height;
			Size() : Width(0),Height(0) {}
			Size(float width, float height) : Width(width), Height(height) {}
			bool operator == (const Size & rhs) const
			{
				return Width == rhs.Width && Height == rhs.Height;
			}
			bool operator != (const Size& rhs) const { return !((*this) == rhs); }
		};

		struct Rect
		{
			Rect() : X(0),Y(0),Width(0),Height(0)
			{}
			Rect(float x, float y, float width, float height)
				: X(x), Y(y), Width(width), Height(height)
			{}

			float X, Y, Width, Height;

			float Top() const { return Y; }
			float Left() const { return X; }
			float Bottom() const { return Y + Height; }
			float Right() const { return X + Width; }

			bool IsEmpty() const { return Height * Width <= .0f; }
			static Rect Empty;
		};
#endif

		void SetLogicalSize(Size logicalSize);

		void SetCurrentOrientation(DisplayOrientations currentOrientation);
		void SetCurrentOrientation(DirectX::FXMVECTOR quaternion);
		void SetDpi(float dpi);
		void SetCompositionScale(float compositionScaleX, float compositionScaleY);

		void ValidateDevice();
		void HandleDeviceLost();
		void RegisterDeviceNotify(IDeviceNotify* deviceNotify);
		void Trim();
		void Present();

		UINT GetMultiSampleCount() const { return m_multiSampleLevel; }
		UINT GetMultiSampleQuality() const { return m_multiSampleQuality; }

		DeviceHostType			GetDeviceHostType() const				{ return m_deviceHostType; }

		RenderTarget&			GetBackBufferRenderTarget()				{ return m_BackBuffer; }

		// Device Accessors.
		Size					GetOutputSize() const					{ return m_outputSize; }
		Size					GetLogicalSize() const				{ return m_logicalSize; }

		// D3D Accessors.
		ID3D11Device2*			GetD3DDevice() const override			{ return m_d3dDevice.Get(); }
		ID3D11DeviceContext2*	GetD3DDeviceContext() const	override	{ return m_d3dContext.Get(); }
		std::mutex&				GetD3DContextMutext() const				{ return m_d3dMutex; }
		IDXGISwapChain1*		GetSwapChain() const					{ return m_swapChain.Get(); }
		D3D_FEATURE_LEVEL		GetDeviceFeatureLevel() const			{ return m_d3dFeatureLevel; }

		ID3D11RenderTargetView*	GetBackBufferRenderTargetView() const	{ return m_d3dRenderTargetView.Get(); }
		ID3D11DepthStencilView* GetDepthStencilView() const				{ return m_d3dDepthStencilView.Get(); }

		D3D11_VIEWPORT			GetScreenViewport() const				{ return m_screenViewport; }
		XMMATRIX				GetOrientationTransform3D() const		{ return XMLoadFloat4x4(&m_orientationTransform3D); }

		// D2D Accessors.
		ID2D1Factory2*			GetD2DFactory() const override			{ return m_d2dFactory.Get(); }
		ID2D1Device1*			GetD2DDevice() const override			{ return m_d2dDevice.Get(); }
		ID2D1DeviceContext1*	GetD2DDeviceContext() const	override	{ return m_d2dContext.Get(); }
		ID2D1Bitmap1*			GetD2DTargetBitmap() const				{ return m_d2dTargetBitmap.Get(); }
		IDWriteFactory2*		GetDWriteFactory() const override		{ return m_dwriteFactory.Get();	 }
		//IWICImagingFactory2*	GetWicImagingFactory() const			{ return m_wicFactory.Get(); }
		D2D1::Matrix3x2F		GetOrientationTransform2D() const		{ return m_orientationTransform2D; }

	private:
		void CreateDeviceIndependentResources();
		void CreateDeviceResources();
		void CreateWindowSizeDependentResources();

		void CreateSwapChainForComposition(IDXGIFactory2* dxgiFactory, DXGI_SWAP_CHAIN_DESC1 *pSwapChainDesc);

		DXGI_MODE_ROTATION ComputeDisplayRotation();

		// Direct3D objects.
		Microsoft::WRL::ComPtr<ID3D11Device2>			m_d3dDevice;
		Microsoft::WRL::ComPtr<ID3D11DeviceContext2>	m_d3dContext;
		Microsoft::WRL::ComPtr<IDXGISwapChain1>			m_swapChain;
		// Lockes when Device or Immediente context is busy
		mutable std::mutex								m_d3dMutex;

		// Direct3D rendering objects. Required for 3D.
		Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	m_d3dRenderTargetView;
		Microsoft::WRL::ComPtr<ID3D11DepthStencilView>	m_d3dDepthStencilView;
		D3D11_VIEWPORT									m_screenViewport;

		RenderableTexture2D								m_ColorBackBuffer;
		DepthStencilBuffer								m_DepthBuffer;
		RenderTarget									m_BackBuffer;

		// Direct2D drawing components.
		Microsoft::WRL::ComPtr<ID2D1Factory2>			m_d2dFactory;
		Microsoft::WRL::ComPtr<ID2D1Device1>			m_d2dDevice;
		Microsoft::WRL::ComPtr<ID2D1DeviceContext1>		m_d2dContext;
		Microsoft::WRL::ComPtr<ID2D1Bitmap1>			m_d2dTargetBitmap;

		// DirectWrite drawing components.
		Microsoft::WRL::ComPtr<IDWriteFactory2>			m_dwriteFactory;
		//Microsoft::WRL::ComPtr<IWICImagingFactory2>		m_wicFactory;

		//std::unique_ptr<DirectX::EffectFactory>		   m_EffectFactory;
		DeviceHostType								   m_deviceHostType;

#if defined (__cplusplus_winrt)
		// Cached reference to the XAML panel.
		Windows::UI::Xaml::Controls::SwapChainPanel^   m_swapChainPanel;
		// Cached reference to the Window.
		Platform::Agile<Windows::UI::Core::CoreWindow> m_window;
#endif
		HWND										   m_hWnd;

		// Cached device properties.
		D3D_FEATURE_LEVEL								m_d3dFeatureLevel;
		Size											m_d3dRenderTargetSize;
		Size											m_outputSize;
		Size											m_logicalSize;
		DisplayOrientations								m_nativeOrientation;
		DisplayOrientations								m_currentOrientation;
		float											m_dpi;
		float											m_dpiScaleX;
		float											m_dpiScaleY;
		UINT											m_multiSampleLevel;
		UINT											m_multiSampleQuality;
		// Transforms used for display orientation.
		D2D1::Matrix3x2F								m_orientationTransform2D;
		DirectX::XMFLOAT4X4								m_orientationTransform3D;

		// The IDeviceNotify can be held directly as it owns the DeviceResources.
		IDeviceNotify* m_deviceNotify;
	};

	class GraphicsResource
	{
	public:
		virtual ~GraphicsResource();

		void SetDeviceResource(const std::shared_ptr<DeviceResources> pDeviceResources);

		virtual void CreateDeviceResources() = 0;
		virtual void CreateSizeDependentResource() = 0;

	protected:
		std::shared_ptr<DeviceResources> m_pDeviceResources;
	};
}