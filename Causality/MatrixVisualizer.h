#pragma once
#include "FloatHud.h"
#include "SmartPointers.h"
#include <Eigen/Core>

namespace Causality
{
	enum class MatrixVisualizerColorMode
	{
		GrayScale = 0,
		RGB = 1,
		PersudoColor = 2,
	};


	class MatrixVisualizer : public SpriteObject
	{
	public:
		typedef Eigen::MatrixXf MatrixType;
		typedef MatrixVisualizerColorMode ColorMode;

		void Create(IRenderDevice* pDevice, size_t rows, size_t cols);
		void SetColorMode(ColorMode mode);
		void SetAutoContrast(bool autoContrast);

		void UpdateTexture(IRenderContext* pContext, const MatrixType& matrix);

	private:
		bool		m_autoContrast;
		ColorMode	m_colorMode;
		uptr<DynamicTexture2D> m_pDTexture;
	};

}