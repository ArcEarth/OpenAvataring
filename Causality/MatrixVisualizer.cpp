#include "pch_bcl.h"
#include "MatrixVisualizer.h"
#include <Textures.h>

using namespace Causality;
using namespace DirectX;

DXGI_FORMAT DXGIFormatFromColorMode(MatrixVisualizerColorMode mode)
{
	return DXGI_FORMAT_R32_FLOAT;
}

void MatrixVisualizer::Create(IRenderDevice * pDevice, size_t rows, size_t cols)
{
	DXGI_FORMAT format = DXGIFormatFromColorMode(m_colorMode);
	m_pDTexture.reset(new DynamicTexture2D(pDevice, rows, cols, DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT));
}

inline void MatrixVisualizer::UpdateTexture(IRenderContext * pContext, const MatrixType& matrix)
{
	m_pDTexture->SetData(pContext, matrix.data());
}
