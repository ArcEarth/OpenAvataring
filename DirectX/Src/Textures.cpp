#include "pch_directX.h"
#include "Textures.h"
#include "DirectXHelper.h"
#include "DXGIFormatHelper.h"
#include <DDSTextureLoader.h>
#include <WICTextureLoader.h>
#include <algorithm>
#include <ScreenGrab.h>
#include <wincodec.h>
#include <d2d1_1.h>

using namespace DirectX;
using namespace std;

bool IsDXGIFormatD2DSupported(DXGI_FORMAT format)
{
	return format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_A8_UNORM || format == DXGI_FORMAT_B8G8R8X8_UNORM || format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC2_UNORM || format == DXGI_FORMAT_BC3_UNORM;
}

Texture::Texture(Texture&& Src)
	: m_pResource(std::move(Src.m_pResource))
	, m_pShaderResourceView(std::move(Src.ShaderResourceView()))
{
}

Texture& Texture::operator = (Texture&& Src)
{
	m_pResource = std::move(Src.m_pResource);
	m_pShaderResourceView = std::move(Src.ShaderResourceView());
	return *this;
}

Texture2D::Texture2D(Texture2D &&Src)
	:DirectX::Texture(std::move(static_cast<DirectX::Texture&>(Src)))
	, m_pTexture(std::move(Src.m_pTexture))
	, m_Description(Src.m_Description)
{
}

Texture2D& Texture2D::operator = (Texture2D &&Src)
{
	Texture::operator=(std::move(Src));
	m_pTexture = Src.m_pTexture;
	m_Description = Src.m_Description;
	return *this;
}


Texture2D::Texture2D(_In_ ID3D11Device* pDevice, _In_ unsigned int Width, _In_ unsigned int Height,
	_In_opt_ unsigned int MipMapLevel,
	_In_opt_ DXGI_FORMAT format,
	_In_opt_ D3D11_USAGE usage,
	_In_opt_ unsigned int bindFlags,
	_In_opt_ unsigned int cpuAccessFlags,
	_In_opt_ unsigned int miscFlags,
	_In_opt_ unsigned int multisamplesCount,
	_In_opt_ unsigned int multisamplesQuality
	)
{
	if (format == DXGI_FORMAT_UNKNOWN) // Invaliad arg
		return;

	if (multisamplesCount > 1)
		bindFlags &= ~D3D11_BIND_SHADER_RESOURCE; // Shader resources is not valiad for MSAA texture

	D3D11_TEXTURE2D_DESC &TextureDesc = m_Description;
	TextureDesc.Width = Width;
	TextureDesc.Height = Height;
	TextureDesc.MipLevels = MipMapLevel;
	TextureDesc.ArraySize = 1;
	TextureDesc.Format = format;
	TextureDesc.SampleDesc.Count = multisamplesCount;
	TextureDesc.SampleDesc.Quality = multisamplesQuality;
	TextureDesc.Usage = usage;
	TextureDesc.BindFlags = bindFlags;
	TextureDesc.CPUAccessFlags = cpuAccessFlags;
	TextureDesc.MiscFlags = miscFlags;
	if (miscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE)
		TextureDesc.ArraySize = 6;

	HRESULT hr = pDevice->CreateTexture2D(&TextureDesc, NULL, &m_pTexture);
	ThrowIfFailed(hr);
	hr = m_pTexture.As<ID3D11Resource>(&m_pResource);
	ThrowIfFailed(hr);

	if ((bindFlags & D3D11_BIND_DEPTH_STENCIL) || !(bindFlags & D3D11_BIND_SHADER_RESOURCE) || DXGIFormatTraits::IsTypeless(format))
	{
		m_pShaderResourceView = nullptr;
		return;
	}

	D3D11_SRV_DIMENSION srvDim = D3D11_SRV_DIMENSION_TEXTURE2D;
	if (miscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE)
		srvDim = D3D11_SRV_DIMENSION_TEXTURECUBE;
	CD3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc(m_pTexture.Get(), srvDim);

	hr = pDevice->CreateShaderResourceView(m_pResource.Get(), &SRVDesc, &m_pShaderResourceView);
	ThrowIfFailed(hr);
}

Texture2D::Texture2D(ID3D11Device * pDevice, const wchar_t * szFileName, ID3D11DeviceContext * pDeviceContext, size_t maxsize, D3D11_USAGE usage, unsigned int bindFlags, unsigned int cpuAccessFlags, unsigned int miscFlags, bool forceSRGB)
{
	this->CreateFromWICFile(pDevice, pDeviceContext, szFileName, maxsize, usage, bindFlags, cpuAccessFlags, forceSRGB);
}

Texture2D::Texture2D(ID3D11Texture2D* pTexture, ID3D11ShaderResourceView* pResourceView)
{
	assert(pTexture);
	m_pTexture = pTexture;
	HRESULT hr = m_pTexture.As<ID3D11Resource>(&m_pResource);
	ThrowIfFailed(hr);

	m_pTexture->GetDesc(&m_Description);
	m_pShaderResourceView = pResourceView;
}

Texture2D::Texture2D(ID3D11ShaderResourceView* pResourceView)
{
	ComPtr<ID3D11Resource> pResource;
	pResourceView->GetResource(&pResource);
	HRESULT hr = pResource.As(&m_pTexture);
	ThrowIfFailed(hr);
	m_pTexture->GetDesc(&m_Description);
	m_pShaderResourceView = pResourceView;
	m_pResource = std::move(pResource);
}

void Texture2D::CopyFrom(ID3D11DeviceContext *pContext, const Texture2D* pSource)
{
	assert(this != pSource);
	assert(this->Resource() != pSource->Resource());
	pContext->CopyResource(this->Resource(), pSource->Resource());
}

ID2D1Bitmap1* RenderableTexture2D::CreateD2DBitmapView(ID2D1DeviceContext *pContext, float dpi)
{
	if (m_pD2dBitmap != nullptr)
		return m_pD2dBitmap;

	ID2D1Bitmap1* bitmap = nullptr;
	// D2D requires BGRA color order
	if (!IsDXGIFormatD2DSupported(Format()))
		throw runtime_error("D2D only support DXGI Format DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_A8_UNORM");

	// Create a Direct2D target bitmap associated with the
	// swap chain back buffer and set it as the current target.
	D2D1_BITMAP_PROPERTIES1 bitmapProperties =
		D2D1::BitmapProperties1(
			D2D1_BITMAP_OPTIONS_TARGET
			| ((m_Description.CPUAccessFlags & D3D11_CPU_ACCESS_READ) ? D2D1_BITMAP_OPTIONS_CPU_READ : D2D1_BITMAP_OPTIONS_NONE),
			D2D1::PixelFormat(Format(), D2D1_ALPHA_MODE_PREMULTIPLIED),
			dpi, dpi
			);

	ComPtr<IDXGISurface2> dxgiSurface;
	ThrowIfFailed(
		m_pTexture.As(&dxgiSurface)
	);

	ThrowIfFailed(
		pContext->CreateBitmapFromDxgiSurface(
			dxgiSurface.Get(),
			&bitmapProperties,
			&bitmap
			)
		);

	m_pD2dBitmap = bitmap;
	return bitmap;
}



void * DynamicTexture2D::Map(ID3D11DeviceContext * pContext)
{
	D3D11_MAPPED_SUBRESOURCE Resouce;
	auto hr = pContext->Map(m_pResource.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &Resouce);
	ThrowIfFailed(hr);
	return Resouce.pData;
}

void DynamicTexture2D::Unmap(ID3D11DeviceContext * pContext)
{
	pContext->Unmap(m_pResource.Get(),0);
}

void DynamicTexture2D::SetData(ID3D11DeviceContext* pContext, const void* Raw_Data, size_t element_size)
{
	assert(element_size == 0 || element_size == DXGIFormatTraits::SizeofDXGIFormatInBytes(Format()));

	if (element_size == 0)
		element_size = DXGIFormatTraits::SizeofDXGIFormatInBytes(Format());


	D3D11_MAPPED_SUBRESOURCE Resouce;
	auto hr = pContext->Map(m_pResource.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &Resouce);
	ThrowIfFailed(hr);
	int srcRowPitch = element_size * Width();
	if (Resouce.RowPitch == srcRowPitch)
		memcpy(Resouce.pData, Raw_Data, element_size * Width() * Height());
	else
	{
		for (int r = 0; r < Height(); r++)
		{
			memcpy((char *)(Resouce.pData) + r *  Resouce.RowPitch,
				   (const char *)(Raw_Data) + r * srcRowPitch,
				   srcRowPitch);
		}
	}
	pContext->Unmap(m_pResource.Get(), 0);
}

DynamicTexture2D::DynamicTexture2D(_In_ ID3D11Device* pDevice, _In_ unsigned int Width, _In_ unsigned int Height, _In_opt_ DXGI_FORMAT Format)
	: Texture2D(pDevice, Width, Height, 1, Format, D3D11_USAGE_DYNAMIC, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE)
{
}

RenderableTexture2D::RenderableTexture2D(_In_ ID3D11Device* pDevice, _In_ unsigned int Width, _In_ unsigned int Height,
	_In_opt_ DXGI_FORMAT Format, _In_opt_ UINT MultiSampleCount, _In_opt_ UINT MultiSampleQuality, _In_opt_ bool Shared)
	: Texture2D(pDevice, Width, Height, 1, Format, D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET | (MultiSampleCount == 1 ? D3D11_BIND_SHADER_RESOURCE : 0), 0, Shared ? D3D11_RESOURCE_MISC_SHARED : 0, MultiSampleCount, MultiSampleQuality)
{
	CD3D11_RENDER_TARGET_VIEW_DESC	RenderTargetViewDesc(m_pTexture.Get(), D3D11_RTV_DIMENSION_TEXTURE2D);
	// Setup the description of the render target view.
	//RenderTargetViewDesc.Format = Format;
	//RenderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	//RenderTargetViewDesc.Texture2D.MipSlice = 0;

	// Create the render target view.
	HRESULT hr = pDevice->CreateRenderTargetView(m_pResource.Get(), nullptr, &m_pRenderTargetView);
	ThrowIfFailed(hr);

	m_pD2dBitmap = nullptr;
}

RenderableTexture2D::RenderableTexture2D(ID3D11Texture2D* pTexture, ID3D11RenderTargetView* pRenderTargetView, ID3D11ShaderResourceView* pShaderResouceView)
	: Texture2D(pTexture, pShaderResouceView)
{
	assert(pRenderTargetView);
	m_pRenderTargetView = pRenderTargetView;
	m_pD2dBitmap = nullptr;
}

inline RenderableTexture2D::RenderableTexture2D(ID3D11RenderTargetView * pRenderTargetView)
{
	m_pRenderTargetView = pRenderTargetView;
	pRenderTargetView->GetResource(&m_pResource);
	m_pResource.As(&m_pTexture);
	m_pTexture->GetDesc(&m_Description);
	m_pD2dBitmap = nullptr;
}

RenderableTexture2D::RenderableTexture2D()
{
	m_pD2dBitmap = nullptr;
}

RenderableTexture2D::RenderableTexture2D(RenderableTexture2D &&source)
	: Texture2D(std::move(source)),
	m_pRenderTargetView(std::move(source.m_pRenderTargetView))
{
	m_pD2dBitmap = source.m_pD2dBitmap;
	source.m_pD2dBitmap = nullptr;
}

RenderableTexture2D& RenderableTexture2D::operator=(RenderableTexture2D &&source)
{
	Texture2D::operator=(std::move(source));
	m_pRenderTargetView = std::move(source.m_pRenderTargetView);
	if (m_pD2dBitmap)
	{
		m_pD2dBitmap->Release();
		m_pD2dBitmap = nullptr;
	}
	m_pD2dBitmap = source.m_pD2dBitmap;
	source.m_pD2dBitmap = nullptr;

	return *this;
}

RenderableTexture2D& RenderableTexture2D::operator=(const RenderableTexture2D &rhs)
{
	Texture2D::operator=(rhs);
	m_pRenderTargetView = rhs.m_pRenderTargetView;
	m_pD2dBitmap = rhs.m_pD2dBitmap;
	if (m_pD2dBitmap)
		m_pD2dBitmap->AddRef();
	return *this;
}

RenderableTexture2D::RenderableTexture2D(const RenderableTexture2D & rhs)
{
	*this = rhs;
}

DepthStencilBuffer::DepthStencilBuffer()
{}
DepthStencilBuffer::DepthStencilBuffer(DepthStencilBuffer&&source)
	: Texture2D(std::move(source)),
	m_pDepthStencilView(std::move(source.m_pDepthStencilView))
{
}
DepthStencilBuffer& DepthStencilBuffer::operator = (DepthStencilBuffer&&source)
{
	Texture2D::operator=(std::move(source));
	m_pDepthStencilView = std::move(source.m_pDepthStencilView);
	return *this;
}



DepthStencilBuffer::DepthStencilBuffer(ID3D11Texture2D * pTexture, ID3D11DepthStencilView * pDSV)
	: Texture2D(pTexture, nullptr)
{
	m_pDepthStencilView = pDSV;
}

DepthStencilBuffer::DepthStencilBuffer(ID3D11DepthStencilView * pDSV)
{
	m_pDepthStencilView = pDSV;
	m_pDepthStencilView->GetResource(&m_pResource);
	m_pResource.As(&m_pTexture);
	m_pTexture->GetDesc(&m_Description);
}

DepthStencilBuffer::DepthStencilBuffer(ID3D11Device* pDevice, unsigned int Width, unsigned int Height, _In_opt_ DXGI_FORMAT Format, _In_opt_ UINT MultiSampleCount, _In_opt_ UINT MultiSampleQuality)
	: Texture2D(pDevice, Width, Height, 1, MultiSampleCount > 1 ? Format : DXGIConvertFormatDSVToResource(Format), D3D11_USAGE_DEFAULT, D3D11_BIND_DEPTH_STENCIL | (MultiSampleCount > 1 ? 0 : D3D11_BIND_SHADER_RESOURCE), 0, 0, MultiSampleCount, MultiSampleQuality)
{
	auto pTexture = m_pResource.Get();
	if (pTexture == nullptr)
		return;
	// Initialize the depth stencil view.

	CD3D11_DEPTH_STENCIL_VIEW_DESC DSVDesc(D3D11_DSV_DIMENSION_TEXTURE2D, Format);
	// Create the depth stencil view.
	HRESULT hr = pDevice->CreateDepthStencilView(pTexture, MultiSampleCount > 1 ? NULL : &DSVDesc, m_pDepthStencilView.GetAddressOf());
	ThrowIfFailed(hr);

	if (MultiSampleCount <= 1)
	{
		CD3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc(m_pTexture.Get(), D3D11_SRV_DIMENSION_TEXTURE2D, DXGIConvertFormatDSVToSRV(Format));
		// Create shader resources view
		hr = pDevice->CreateShaderResourceView(m_pResource.Get(), &SRVDesc, &m_pShaderResourceView);
		ThrowIfFailed(hr);
	}
}

Texture::Texture()
{}

Texture::Texture(ID3D11Resource * pResource, ID3D11ShaderResourceView * pResourceView)
{
	m_pResource = pResource;
	m_pShaderResourceView = pResourceView;
}

Texture::~Texture()
{}

Texture2D::Texture2D()
{}
Texture2D::~Texture2D()
{}

DynamicTexture2D::~DynamicTexture2D()
{}

RenderableTexture2D::~RenderableTexture2D()
{
	if (m_pD2dBitmap)
		m_pD2dBitmap->Release();
}

void RenderableTexture2D::Clear(ID3D11DeviceContext * pDeviceContext, FXMVECTOR Color)
{
	XMFLOAT4A col;
	XMStoreFloat4A(&col, Color);
	if (m_pRenderTargetView)
		pDeviceContext->ClearRenderTargetView(m_pRenderTargetView.Get(), &col.x);
}

DepthStencilBuffer::~DepthStencilBuffer()
{}

Texture* Texture::CreateFromDDSFile(_In_ ID3D11Device* pDevice, _In_z_ const wchar_t* szFileName,
	_In_opt_ size_t maxsize,
	_In_opt_ D3D11_USAGE usage,
	_In_opt_ unsigned int bindFlags,
	_In_opt_ unsigned int cpuAccessFlags,
	_In_opt_ unsigned int miscFlags,
	_In_opt_ bool forceSRGB
	)
{
	HRESULT hr;
	wstring exName(szFileName);
	exName = exName.substr(exName.find_last_of(L'.') + 1);
	std::transform(exName.begin(), exName.end(), exName.begin(), ::towupper);

	ID3D11Resource* pResource;
	ID3D11ShaderResourceView *pView;
	if (exName == L"DDS")
	{
		hr = CreateDDSTextureFromFileEx(pDevice, szFileName, maxsize, usage, bindFlags, cpuAccessFlags, miscFlags, forceSRGB, &pResource, &pView);

		if (FAILED(hr))
			return nullptr;
		// It's an cube texture
		if (miscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE)
		{
			EnvironmentTexture* texture = new EnvironmentTexture(pResource, pView);
			return texture;
		}

		D3D11_RESOURCE_DIMENSION dimension;
		pResource->GetType(&dimension);

		switch (dimension)
		{
		case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
			break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
		{
			Texture2D *texture2D = new Texture2D(pView);
			pResource->Release();
			pView->Release();
			return texture2D;
		}
		break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
			break;
		case D3D11_RESOURCE_DIMENSION_BUFFER:
		case D3D11_RESOURCE_DIMENSION_UNKNOWN:
		default:
			break;
		}
	}
	return nullptr;
}

void Texture::SaveAsDDSFile(_In_ ID3D11DeviceContext *pDeviceContext, _In_z_ const wchar_t* szFileName)
{
	HRESULT hr = SaveDDSTextureToFile(pDeviceContext, m_pResource.Get(), szFileName);
	ThrowIfFailed(hr);
}

bool Texture2D::CreateFromWICFile(_In_ ID3D11Device* pDevice,
	_In_ ID3D11DeviceContext* pDeviceContext,
	_In_z_ const wchar_t* szFileName,
	_In_opt_ size_t maxsize,
	_In_opt_ D3D11_USAGE usage,
	_In_opt_ unsigned int bindFlags,
	_In_opt_ unsigned int cpuAccessFlags,
	_In_opt_ unsigned int miscFlags,
	_In_opt_ bool forceSRGB
	)
{
	HRESULT hr;
	wstring exName(szFileName);
	exName = exName.substr(exName.find_last_of(L'.') + 1);
	std::transform(exName.begin(), exName.end(), exName.begin(), ::towupper);
	Texture2D& texture = *this;
	if (exName == L"DDS")
	{
		hr = CreateDDSTextureFromFileEx(pDevice, szFileName, maxsize, usage, bindFlags, cpuAccessFlags, miscFlags, forceSRGB, &texture.m_pResource, &texture.m_pShaderResourceView);
		ThrowIfFailed(hr);
		D3D11_RESOURCE_DIMENSION dimension;
		texture.m_pResource->GetType(&dimension);
		if (dimension == D3D11_RESOURCE_DIMENSION::D3D11_RESOURCE_DIMENSION_TEXTURE2D)
		{
			ID3D11Texture2D* pTexInterface;
			texture.Resource()->QueryInterface<ID3D11Texture2D>(&pTexInterface);
			pTexInterface->GetDesc(&texture.m_Description);
		}
		else
		{
			throw new exception("Unsupported Format");
		}
		return true;
	}
	else
	{
		hr = CreateWICTextureFromFileEx(pDevice, pDeviceContext, szFileName, maxsize, usage, bindFlags, cpuAccessFlags, miscFlags, forceSRGB, &texture.m_pResource, &texture.m_pShaderResourceView);
		ThrowIfFailed(hr);
		ID3D11Texture2D* pTexInterface;
		texture.Resource()->QueryInterface<ID3D11Texture2D>(&pTexInterface);
		pTexInterface->GetDesc(&texture.m_Description);
		return true;
	}
	return false;
}


bool Texture2D::CreateFromWICMemory(_In_ ID3D11Device* pDevice,
	_In_ ID3D11DeviceContext* pDeviceContext,
	_In_reads_bytes_(wicDataSize) const uint8_t* wicData,
	_In_ size_t wicDataSize,
	_In_opt_ size_t maxsize,
	_In_opt_ D3D11_USAGE usage,
	_In_opt_ unsigned int bindFlags,
	_In_opt_ unsigned int cpuAccessFlags,
	_In_opt_ unsigned int miscFlags,
	_In_opt_ bool forceSRGB
	)
{
	return false;
}

void Texture2D::SaveAsWICFile(ID3D11DeviceContext * pDeviceContext, FileFormat format, const wchar_t * szFileName)
{
	static const GUID containerFormats[] = 
	{
		GUID_ContainerFormatBmp,
		GUID_ContainerFormatPng,
		GUID_ContainerFormatIco,
		GUID_ContainerFormatJpeg,
		GUID_ContainerFormatTiff,
		GUID_ContainerFormatGif,
		GUID_ContainerFormatWmp,
	};

	SaveWICTextureToFile(pDeviceContext, m_pTexture.Get(), containerFormats[format], szFileName);
}

//--------------------------------------------------------------------------------------
static DXGI_FORMAT EnsureNotTypeless(DXGI_FORMAT fmt)
{
	// Assumes UNORM or FLOAT; doesn't use UINT or SINT
	switch (fmt)
	{
	case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
	case DXGI_FORMAT_R32G32B32_TYPELESS:    return DXGI_FORMAT_R32G32B32_FLOAT;
	case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_UNORM;
	case DXGI_FORMAT_R32G32_TYPELESS:       return DXGI_FORMAT_R32G32_FLOAT;
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:  return DXGI_FORMAT_R10G10B10A2_UNORM;
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:     return DXGI_FORMAT_R8G8B8A8_UNORM;
	case DXGI_FORMAT_R16G16_TYPELESS:       return DXGI_FORMAT_R16G16_UNORM;
	case DXGI_FORMAT_R32_TYPELESS:          return DXGI_FORMAT_R32_FLOAT;
	case DXGI_FORMAT_R8G8_TYPELESS:         return DXGI_FORMAT_R8G8_UNORM;
	case DXGI_FORMAT_R16_TYPELESS:          return DXGI_FORMAT_R16_UNORM;
	case DXGI_FORMAT_R8_TYPELESS:           return DXGI_FORMAT_R8_UNORM;
	case DXGI_FORMAT_BC1_TYPELESS:          return DXGI_FORMAT_BC1_UNORM;
	case DXGI_FORMAT_BC2_TYPELESS:          return DXGI_FORMAT_BC2_UNORM;
	case DXGI_FORMAT_BC3_TYPELESS:          return DXGI_FORMAT_BC3_UNORM;
	case DXGI_FORMAT_BC4_TYPELESS:          return DXGI_FORMAT_BC4_UNORM;
	case DXGI_FORMAT_BC5_TYPELESS:          return DXGI_FORMAT_BC5_UNORM;
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:     return DXGI_FORMAT_B8G8R8A8_UNORM;
	case DXGI_FORMAT_B8G8R8X8_TYPELESS:     return DXGI_FORMAT_B8G8R8X8_UNORM;
	case DXGI_FORMAT_BC7_TYPELESS:          return DXGI_FORMAT_BC7_UNORM;
	default:                                return fmt;
	}
}

StagingTexture2D::StagingTexture2D(ID3D11DeviceContext* pContext, const Texture2D* pSourceTexture, _In_opt_ unsigned int cpuAccessFlags)
{
	if (!pContext || !pSourceTexture)
		throw std::invalid_argument("DeviceContext or Source Texture is null.");

	auto pSource = pSourceTexture->Resource();

	D3D11_RESOURCE_DIMENSION resType = D3D11_RESOURCE_DIMENSION_UNKNOWN;
	pSource->GetType(&resType);

	if (resType != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
		ThrowIfFailed(HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED));

	ScopedObject<ID3D11Texture2D> pTexture;
	HRESULT hr = pSource->QueryInterface<ID3D11Texture2D>(&pTexture);

	ThrowIfFailed(hr);

	assert(pTexture.Get());

	pTexture->GetDesc(&m_Description);

	ScopedObject<ID3D11Device> d3dDevice;
	pContext->GetDevice(&d3dDevice);

	if (m_Description.SampleDesc.Count > 1)
	{
		// MSAA content must be resolved before being copied to a staging texture
		m_Description.SampleDesc.Count = 1;
		m_Description.SampleDesc.Quality = 0;

		ScopedObject<ID3D11Texture2D> pTemp;
		hr = d3dDevice->CreateTexture2D(&m_Description, 0, &pTemp);
		ThrowIfFailed(hr);

		assert(pTemp.Get());

		DXGI_FORMAT fmt = EnsureNotTypeless(m_Description.Format);

		UINT support = 0;
		hr = d3dDevice->CheckFormatSupport(fmt, &support);
		ThrowIfFailed(hr);

		if (!(support & D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE))
			throw std::runtime_error("Multisampling resolve failed");

		for (UINT item = 0; item < m_Description.ArraySize; ++item)
		{
			for (UINT level = 0; level < m_Description.MipLevels; ++level)
			{
				UINT index = D3D11CalcSubresource(level, item, m_Description.MipLevels);
				pContext->ResolveSubresource(pTemp.Get(), index, pSource, index, fmt);
			}
		}

		m_Description.BindFlags = 0;
		m_Description.MiscFlags &= D3D11_RESOURCE_MISC_TEXTURECUBE;
		m_Description.CPUAccessFlags = cpuAccessFlags;
		m_Description.Usage = D3D11_USAGE_STAGING;

		ID3D11Texture2D* pTex;
		hr = d3dDevice->CreateTexture2D(&m_Description, 0, &pTex);
		ThrowIfFailed(hr);
		assert(m_pResource.Get());
		pContext->CopyResource(m_pResource.Get(), pTemp.Get());
		//m_pTexture = pTexture.Get();
	}
	else if ((m_Description.Usage == D3D11_USAGE_STAGING) && (m_Description.CPUAccessFlags & cpuAccessFlags))
	{
		// Handle case where the source is already a staging texture we can use directly
		m_pResource = pTexture.Get();
	}
	else
	{
		// Otherwise, create a staging texture from the non-MSAA source
		m_Description.BindFlags = 0;
		m_Description.MiscFlags &= D3D11_RESOURCE_MISC_TEXTURECUBE;
		m_Description.CPUAccessFlags = cpuAccessFlags;
		m_Description.Usage = D3D11_USAGE_STAGING;

		ID3D11Texture2D* pTex;
		hr = d3dDevice->CreateTexture2D(&m_Description, 0, &pTex);
		m_pResource = pTex;
		ThrowIfFailed(hr);

		assert(m_pResource.Get());

		pContext->CopyResource(m_pResource.Get(), pSource);
		//m_pTexture = pTexture.Get();
	}
}

//--------------------------------------------------------------------------------------
// Get surface information for a particular format
//--------------------------------------------------------------------------------------
static void GetSurfaceInfo(_In_ size_t width,
	_In_ size_t height,
	_In_ DXGI_FORMAT fmt,
	_Out_opt_ size_t* outNumBytes,
	_Out_opt_ size_t* outRowBytes,
	_Out_opt_ size_t* outNumRows)
{
	size_t numBytes = 0;
	size_t rowBytes = 0;
	size_t numRows = 0;

	bool bc = false;
	bool packed = false;
	size_t bcnumBytesPerBlock = 0;
	switch (fmt)
	{
	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC1_UNORM:
	case DXGI_FORMAT_BC1_UNORM_SRGB:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC4_UNORM:
	case DXGI_FORMAT_BC4_SNORM:
		bc = true;
		bcnumBytesPerBlock = 8;
		break;

	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC6H_UF16:
	case DXGI_FORMAT_BC6H_SF16:
	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
		bc = true;
		bcnumBytesPerBlock = 16;
		break;

	case DXGI_FORMAT_R8G8_B8G8_UNORM:
	case DXGI_FORMAT_G8R8_G8B8_UNORM:
		packed = true;
		break;
	}

	if (bc)
	{
		size_t numBlocksWide = 0;
		if (width > 0)
		{
			numBlocksWide = std::max<size_t>(1, (width + 3) / 4);
		}
		size_t numBlocksHigh = 0;
		if (height > 0)
		{
			numBlocksHigh = std::max<size_t>(1, (height + 3) / 4);
		}
		rowBytes = numBlocksWide * bcnumBytesPerBlock;
		numRows = numBlocksHigh;
	}
	else if (packed)
	{
		rowBytes = ((width + 1) >> 1) * 4;
		numRows = height;
	}
	else
	{
		size_t bpp = DXGIFormatTraits::SizeofDXGIFormatInBits(fmt);
		rowBytes = (width * bpp + 7) / 8; // round up to nearest byte
		numRows = height;
	}

	numBytes = rowBytes * numRows;
	if (outNumBytes)
	{
		*outNumBytes = numBytes;
	}
	if (outRowBytes)
	{
		*outRowBytes = rowBytes;
	}
	if (outNumRows)
	{
		*outNumRows = numRows;
	}
}

std::unique_ptr<uint8_t> StagingTexture2D::GetData(ID3D11DeviceContext *pContext)
{
	size_t rowPitch, slicePitch, rowCount;
	auto & desc = m_Description;
	GetSurfaceInfo(desc.Width, desc.Height, desc.Format, &slicePitch, &rowPitch, &rowCount);

	// Setup pixels
	std::unique_ptr<uint8_t> pixels(new (std::nothrow) uint8_t[slicePitch]);
	if (!pixels)
		throw std::bad_alloc();

	D3D11_MAPPED_SUBRESOURCE mapped;

	HRESULT hr = pContext->Map(m_pResource.Get(), 0, D3D11_MAP_READ, 0, &mapped);
	ThrowIfFailed(hr);

	const uint8_t* sptr = reinterpret_cast<const uint8_t*>(mapped.pData);
	if (!sptr)
	{
		pContext->Unmap(m_pResource.Get(), 0);
		throw std::runtime_error("Resource is invalid.");
	}

	uint8_t* dptr = pixels.get();

	for (size_t h = 0; h < rowCount; ++h)
	{
		size_t msize = std::min<size_t>(rowPitch, mapped.RowPitch);
		memcpy_s(dptr, rowPitch, sptr, msize);
		sptr += mapped.RowPitch;
		dptr += rowPitch;
	}

	pContext->Unmap(m_pResource.Get(), 0);

	return pixels;
}

StagingTexture2D::StagingTexture2D(ID3D11Texture2D* pTexture)
	: Texture2D(pTexture)
{
}

RenderTarget::RenderTarget() {
	m_Viewport.TopLeftX = 0;
	m_Viewport.TopLeftY = 0;
	m_Viewport.Width = 0;
	m_Viewport.Height = 0;
	m_Viewport.MinDepth = D3D11_MIN_DEPTH;
	m_Viewport.MaxDepth = D3D11_MAX_DEPTH;
}

RenderTarget::RenderTarget(RenderableTexture2D & colorBuffer, DepthStencilBuffer & dsBuffer)
	:m_ColorBuffer(colorBuffer), m_DepthStencilBuffer(dsBuffer)
{
	m_Viewport.TopLeftX = 0;
	m_Viewport.TopLeftY = 0;
	m_Viewport.Width = (float)m_ColorBuffer.Width();
	m_Viewport.Height = (float)m_ColorBuffer.Height();
	m_Viewport.MinDepth = D3D11_MIN_DEPTH;
	m_Viewport.MaxDepth = D3D11_MAX_DEPTH;
}

RenderTarget::RenderTarget(RenderableTexture2D & colorBuffer, DepthStencilBuffer & dsBuffer, const D3D11_VIEWPORT & viewPort)
	: m_ColorBuffer(colorBuffer), m_DepthStencilBuffer(dsBuffer), m_Viewport(viewPort)
{}

RenderTarget::RenderTarget(ID3D11Device * pDevice, size_t width, size_t height ,_In_opt_ DXGI_FORMAT colorFormat , _In_opt_ DXGI_FORMAT depthFormat )
	: m_ColorBuffer(pDevice, width, height, colorFormat), m_DepthStencilBuffer(pDevice, width, height, depthFormat)
{
	m_Viewport.TopLeftX = 0;
	m_Viewport.TopLeftY = 0;
	m_Viewport.Width = (float)width;
	m_Viewport.Height = (float)height;
	m_Viewport.MinDepth = D3D11_MIN_DEPTH;
	m_Viewport.MaxDepth = D3D11_MAX_DEPTH;
}

RenderableTexture2D::RenderableTexture2D(IDXGISwapChain * pSwapChain)
{
	ThrowIfFailed(
		pSwapChain->GetBuffer(0, IID_PPV_ARGS(&m_pTexture))
		);
	
	ComPtr<ID3D11Device> pDevice;
	pSwapChain->GetDevice(IID_PPV_ARGS(&pDevice));
	// Create render target view.
	ThrowIfFailed(
		pDevice->CreateRenderTargetView(m_pTexture.Get(), nullptr, &m_pRenderTargetView)
		);
}

RenderTarget RenderTarget::Subview(const D3D11_VIEWPORT & viewport)
{
	RenderTarget target(m_ColorBuffer, m_DepthStencilBuffer, viewport);
	return target;
}

/// <summary>
/// return a Const reference to the ViewPort binding this render target texture for RS stage.
/// </summary>
/// <returns> the return value is default to be (0,0) at left-top corner , while min-max depth is (0,1000) </returns>

const D3D11_VIEWPORT & RenderTarget::ViewPort() const
{
	return m_Viewport;
}

void RenderTarget::Clear(ID3D11DeviceContext * pContext, FXMVECTOR Color)
{
	m_ColorBuffer.Clear(pContext, Color);
	m_DepthStencilBuffer.Clear(pContext);
}

/// <summary>
/// return a Reference to the ViewPort binding this render target texture for RS stage.
/// </summary>
/// <returns> the return value is default to be (0,0) at left-top corner , while min-max depth is (0,1000) </returns>

D3D11_VIEWPORT & RenderTarget::ViewPort()
{
	return m_Viewport;
}

RenderableTexture2D & RenderTarget::ColorBuffer()
{
	return m_ColorBuffer;
}

const RenderableTexture2D & RenderTarget::ColorBuffer() const
{
	return m_ColorBuffer;
}

DepthStencilBuffer & RenderTarget::DepthBuffer()
{
	return m_DepthStencilBuffer;
}

const DepthStencilBuffer & RenderTarget::DepthBuffer() const
{
	return m_DepthStencilBuffer;
}

/// <summary>
/// Sets as render target with default no DepthStencil.
/// </summary>
/// <param name="pDeviceContext">The pointer to device context.</param>
/// <param name="pDepthStencil">The pointer to depth stencil view.</param>

void RenderTarget::SetAsRenderTarget(ID3D11DeviceContext * pDeviceContext)
{
	auto pTargetView = m_ColorBuffer.RenderTargetView();
	pDeviceContext->RSSetViewports(1, &m_Viewport);
	pDeviceContext->OMSetRenderTargets(1, &pTargetView, m_DepthStencilBuffer);
}

CubeTexture::CubeTexture(ID3D11Resource * pResource, ID3D11ShaderResourceView * pResourceView)
	: Texture(pResource, pResourceView)
{
}

void CubeTexture::Initialize(ID3D11Device * pDevice, const std::wstring(&TextureFiles)[6])
{
	for (int i = 0; i < 6; i++)
	{
		ThrowIfFailed(CreateDDSTextureFromFile(pDevice, TextureFiles[i].c_str(), &(m_pTextures[i]), &(m_pTextureView[i])));
	}
}

CubeTexture::CubeTexture(ID3D11Device * pDevice, const std::wstring(&TextureFiles)[6])
{
	Initialize(pDevice, TextureFiles);
}

CubeTexture::CubeTexture()
{}

CubeTexture::~CubeTexture()
{}

ID3D11ShaderResourceView * CubeTexture::at(unsigned int face)
{
	return m_pTextureView[face];
}

ID3D11ShaderResourceView * const * CubeTexture::ResourcesView()
{
	return m_pTextureView;
}

EnvironmentTexture::~EnvironmentTexture()
{

}

EnvironmentTexture::EnvironmentTexture(ID3D11Device * pDevice, unsigned int FaceSize, bool Renderable, DXGI_FORMAT Format)
	: Texture2D(pDevice, FaceSize, FaceSize, 1, Format, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | (Renderable ? D3D11_BIND_RENDER_TARGET : 0), 0, D3D11_RESOURCE_MISC_TEXTURECUBE | (Renderable ? D3D11_RESOURCE_MISC_GENERATE_MIPS : 0))
{
	if (Renderable)
		CreateRenderTargetViewArray(pDevice, Format);
}

EnvironmentTexture::EnvironmentTexture(ID3D11Device * pDevice, const wchar_t * szFileName)
{
	auto pTex = Texture::CreateFromDDSFile(pDevice, szFileName, 64, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, D3D11_RESOURCE_MISC_TEXTURECUBE);
	auto pEtex = static_cast<EnvironmentTexture*>(pTex);
	if (pEtex)
	{
		*this = *pEtex;
		delete pEtex;
	}
}

EnvironmentTexture EnvironmentTexture::CreateFromDDSFile(ID3D11Device * pDevice, const wchar_t * szFileName)
{
	return EnvironmentTexture(pDevice,szFileName);
}

void EnvironmentTexture::GenerateMips(ID3D11DeviceContext * pContext)
{
	pContext->GenerateMips(m_pShaderResourceView.Get());
}

ID3D11RenderTargetView * EnvironmentTexture::RenderTargetView(int face)
{
	return m_pRenderTargetViews[face].Get();
}

ID3D11RenderTargetView * const * EnvironmentTexture::RenderTargetViews()
{
	static_assert(sizeof(ComPtr<ID3D11RenderTargetView>) == sizeof(ID3D11RenderTargetView*), "issue with compiler");
	return m_pRenderTargetViews[0].GetAddressOf();
}

void EnvironmentTexture::CreateRenderTargetViewArray(ID3D11Device * pDevice, DXGI_FORMAT format)
{
	if (format == DXGI_FORMAT_UNKNOWN)
		format = m_Description.Format;
	CD3D11_RENDER_TARGET_VIEW_DESC rtvDesc(D3D11_RTV_DIMENSION_TEXTURE2DARRAY, format, 0, 0, 1);
	for (int i = 0; i < 6; i++) {
		rtvDesc.Texture2DArray.FirstArraySlice = i;
		pDevice->CreateRenderTargetView(m_pResource.Get(), &rtvDesc, &m_pRenderTargetViews[i]);
	}
}

EnvironmentTexture::EnvironmentTexture(ID3D11Resource * pTexture, ID3D11ShaderResourceView * pResourceView)
	: Texture2D(pResourceView)
{
}
