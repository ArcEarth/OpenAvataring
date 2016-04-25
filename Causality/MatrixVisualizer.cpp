#include "pch_bcl.h"
#include "MatrixVisualizer.h"
#include "Scene.h"
#include <Textures.h>


using namespace Causality;
using namespace DirectX;
using namespace DirectX::PackedVector;

static DXGI_FORMAT g_pixelFomat = DXGI_FORMAT::DXGI_FORMAT_B8G8R8A8_UNORM;

DXGI_FORMAT DXGIFormatFromColorMode(MatrixVisualizer::ColorMode mode)
{
	return DXGI_FORMAT_R32_FLOAT;
}

MatrixVisualizer::MatrixVisualizer()
{
	m_dirty = false;
	m_rows = 0;
	m_cols = 0;
	m_autoContrast = true;
	m_colorMode = CM_PersudoColor;
}

MatrixVisualizer::~MatrixVisualizer()
{

}

bool MatrixVisualizer::IsVisible(const BoundingGeometry & viewFrustum) const
{
	auto pVisual = FirstAncesterOfType<IVisual>();
	return m_IsEnabled && pVisual && !empty() && pVisual->IsVisible(viewFrustum);
}

void MatrixVisualizer::Render(IRenderContext * pContext, IEffect * pEffect)
{
	if (m_dirty)
		UpdateTexture(pContext);
	//SpriteObject::Render(pContext, pEffect);
}

void MatrixVisualizer::CreateDeviceResources(IRenderDevice * pDevice, size_t rows, size_t cols)
{
	m_rows = rows; 
	m_cols = cols;

	DXGI_FORMAT format = DXGIFormatFromColorMode(m_colorMode);
	m_pDTexture.reset(new DynamicTexture2D(pDevice, cols, rows, DXGI_FORMAT::DXGI_FORMAT_B8G8R8A8_UNORM));
	SpriteObject::SetTexture(m_pDTexture.get());
}

void MatrixVisualizer::CreateDeviceResources(size_t rows, size_t cols)
{
	auto pDevice = this->Scene->GetRenderDevice();
	CreateDeviceResources(pDevice, rows, cols);
}

void MatrixVisualizer::SetAutoContrast(bool autoContrast)
{
	m_autoContrast = autoContrast;
}

bool MatrixVisualizer::UpdateTexture()
{
	auto pContext = this->Scene->GetRenderContext();
	return UpdateTexture(pContext);
}

bool MatrixVisualizer::UpdateTexture(IRenderContext * pContext)
{
	if (m_dirty && m_cols * m_rows > 0 && (!m_pDTexture || m_pDTexture->Height() != m_rows || m_pDTexture->Width() != m_cols))
	{
		//ComPtr<ID3D11Device> pDevice;
		//pContext->GetDevice(&pDevice);
		CreateDeviceResources(m_rows, m_cols);
	}

	if (m_pDTexture && m_updateMutex.try_lock())
	{
		std::lock_guard<std::mutex> guard(m_updateMutex, std::adopt_lock);

		m_pDTexture->SetData(pContext, m_pixels.data());
		m_dirty = false;
		return true;
	}
	return false;
}

template <class MatrixType>
inline void MatrixVisualizer::UpdateMatrixTemplate(const MatrixType & matrix)
{
	auto bounds = matrix.bounds();

	if (bounds[0] != m_rows || 
		bounds[1] != m_cols)
	{
		m_rows = bounds[0];
		m_cols = bounds[1];
	}

	if (m_pixels.size() != m_rows*m_cols)
		m_pixels.resize(m_rows*m_cols);

	int sz = matrix.size();
	float min = .0f, scale = 1.0f;
	if (m_autoContrast)
	{
		auto pair = std::minmax_element(matrix.data(), matrix.data() + sz);
		min = *pair.first;
		scale = *pair.second;
		scale = 1.0f / (scale - min);
	}

	std::lock_guard<std::mutex> guard(m_updateMutex);

	XMVECTOR vc;
	XMVECTOR vc1_hsv = XMVectorSet(.0,.8f,0.8f,1.0f);

	switch (m_colorMode)
	{
	case CM_GrayScale:
		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < m_cols; j++)
			{
				float t = (matrix(i, j) - min) * scale;

				vc = XMVectorReplicate(t);

				XMStoreColor(&m_pixels[i*m_cols + j],vc);
			}
		}
		break;
	case CM_PersudoColor:
		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < m_cols; j++)
			{
				float t = (matrix(i, j) - min) * scale; // t in [0,1]

				float hue = (1.0f - t) * 2.0 / 3.0;
				XMVECTOR vchsv = XMVectorSetX(vc1_hsv, hue);
				vc = XMColorHSVToRGB(vchsv);

				XMStoreColor(&m_pixels[i*m_cols + j], vc);
			}
		}
		break;
	case CM_RGB:
	case CM_RGBA:
	default:
		assert(!"This color mode are not currently supported");
		break;
	}
	//assert(matrix.size() == m_rows * m_cols && m_pDTexture);
	m_dirty = true;
}

bool MatrixVisualizer::empty() const { 
	return m_pixels.empty(); }

void MatrixVisualizer::UpdateMatrix(const FloatMatrixType & matrix)
{
	UpdateMatrixTemplate<FloatMatrixType>(matrix);
}

void MatrixVisualizer::UpdateMatrix(const DoubleMatrixType & matrix)
{
	UpdateMatrixTemplate<DoubleMatrixType>(matrix);
}
