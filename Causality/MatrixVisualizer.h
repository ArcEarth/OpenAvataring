#pragma once
#include "FloatHud.h"
#include "SmartPointers.h"
#include "String.h"
#include <HUD.h>
#include <mutex>

namespace Causality
{
	class MatrixVisualizer : public SpriteObject
	{
	public:
		enum ColorMode
		{
			CM_GrayScale = 0,
			CM_PersudoColor = 1,
			CM_RGB = 2,
			CM_RGBA = 3,
		};

		typedef gsl::span<const float,-1,-1>	FloatMatrixType;
		typedef gsl::span<const double, -1, -1>	DoubleMatrixType;
		typedef gsl::span<Vector3, -1, -1>  RGBMatrixType;
		typedef gsl::span<Vector4, -1, -1>  RGBAMatrixType;
		//B-G-R-A pixel type if color mode are set
		typedef DirectX::PackedVector::XMCOLOR	  PixelType;

		MatrixVisualizer();
		~MatrixVisualizer() override;

		bool IsVisible(const BoundingGeometry & viewFrustum) const override;

		virtual void Render(IRenderContext * pContext, IEffect* pEffect = nullptr) override;

		void CreateDeviceResources(IRenderDevice* pDevice, size_t rows, size_t cols);
		void CreateDeviceResources(size_t rows, size_t cols);
		void SetColorMode(ColorMode mode);
		void SetAutoContrast(bool autoContrast);

		int	 rows() const { return m_rows; }
		int	 cols() const { return m_cols; }

		bool empty() const;

		void UpdateMatrix(const FloatMatrixType& matrix);
		void UpdateMatrix(const DoubleMatrixType& matrix);
	private:
		template <class MatrixType>
		void UpdateMatrixTemplate(const MatrixType & matrix);
	protected:
		bool UpdateTexture();
		bool UpdateTexture(IRenderContext* pContext);

	private:
		bool					m_autoContrast;
		bool					m_colMajor;
		bool					m_dirty;
		int						m_rows, m_cols;
		ColorMode				m_colorMode;
		uptr<DynamicTexture2D>	m_pDTexture;
		vector<PixelType>		m_pixels;
		std::mutex				m_updateMutex;
	};
}