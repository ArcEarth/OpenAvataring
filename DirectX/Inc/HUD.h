#pragma once

#include <string>
#include <memory>
#include <limits>
#include <dwrite_2.h>
#include <d2d1_2.h>
#include "DirectXMathExtend.h"
#include "Textures.h"

namespace DirectX
{
	/// <summary>
	/// Convert A point in World space to texture space
	/// </summary>
	/// <param name="v">the point's coordinate in projection space</param>
	/// <param name="viewport">layout in order x,y,width,height</param>
	/// <returns></returns>
	inline XMVECTOR XM_CALLCONV XMVector3ConvertToTextureCoord(FXMVECTOR v, FXMVECTOR viewport)
	{
		static XMVECTORF32 conv = { 0.5f,-0.5f,.0f,0.f };
		static XMVECTORF32 addv = { 0.5f, 0.5f,.0f,0.f };
		static XMVECTORU32 g_XMMaskXY = { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 };
		XMVECTOR v1 = conv.v;
		XMVECTOR v2 = addv.v;
		XMVECTOR nv;
		nv = XMVectorMultiplyAdd(v, v1, v2); // convert from [-1,1]x[1,-1] to [0,1]x[0,1] range (Y is fliped)
		v1 = XMVectorAndInt(viewport, g_XMMaskXY.v);
		v2 = XMVectorShiftLeft<2>(viewport, XMVectorZero());
		nv = XMVectorMultiplyAdd(nv, v2, v1); // convert to viewport size and add viewport leff-top
		return nv;
	}

	/// <summary>
	/// Convert A point in World space to texture space
	/// </summary>
	/// <param name="v">the point's coordinate</param>
	/// <param name="viewport">layout in order x,y,width,height</param>
	/// <param name="projection">the worldViewProjection Matrix</param>
	/// <returns></returns>
	inline XMVECTOR XM_CALLCONV XMVector3ConvertToTextureCoord(FXMVECTOR v, FXMVECTOR viewport, CXMMATRIX worldViewProjection)
	{
		XMVECTOR nv = XMVector3TransformCoord(v, worldViewProjection);
		return XMVector3ConvertToTextureCoord(nv, viewport);
	}


	/// <summary>
	/// Convert a texture coord to view-ray direction in view space
	/// </summary>
	/// <param name="v"></param>
	/// <param name="viewport"></param>
	/// <returns></returns>
	inline XMVECTOR XM_CALLCONV XMVector2ConvertToViewDirection(FXMVECTOR v, FXMVECTOR viewport, CXMMATRIX invProjection)
	{
		XMVECTOR nv;
		XMVECTOR vs = XMVectorShiftLeft<2>(viewport, XMVectorZero());
		nv -= viewport;
		nv /= vs;
		nv = XMVectorSelect(g_XMOne.v, nv, g_XMSelect1100.v);
		nv = XMVector3TransformCoord(nv, invProjection);
		nv = XMVector3Normalize(nv);
		return nv;
	}

	namespace Scene
	{
		class TextBlock;

		using Microsoft::WRL::ComPtr;

		enum class HorizentalAlignmentEnum
		{
			Left,
			Right,
			Center,
			Stretch,
		};

		enum class VerticalAlignmentEnum
		{
			Top,
			Bottom,
			Center,
			Stretch,
		};

		class Panel;

		const float AutoSize = std::numeric_limits<float>::signaling_NaN();

		XM_ALIGNATTR
		class HUDElement : public AlignedNew<XMVECTOR>
		{
		public:
			HUDElement();
			virtual ~HUDElement();

			float Width() const;
			float Height() const;
			void  SetWidth(float width);
			void  SetHeight(float height);

			const Vector2& Size() const;
			const Vector2& Position() const;

			void SetSize(const Vector2& size);
			void SetPosition(const Vector2& position);

			float Opticity() const { return m_opticity; }
			void SetOpticity(float opticity);

			bool Visiability() const { return m_visiable; }
			void SetVisiability(bool visiable) { m_visiable = visiable; }

			int	ZIndex() const { return m_zIndex; }

			const HUDElement* Parent() const { return m_parent; }
			HUDElement* Parent() { return m_parent; }

			// Layout interface
			bool IsLayoutUptoDate() const;
			void InvaliadMeasure();
			Vector2 Measure(const Vector2& availableSize);
			Vector2 Arrange(const Vector2& actualSize);
			virtual Vector2 MeasureOverride(const Vector2& availableSize);
			virtual Vector2 ArrangeOverride(const Vector2& actualSize);
			virtual void UpdateLayout();

			// Rendering interface
			virtual void Render(ID2D1DeviceContext* context);

			// Allows Panel to change parent property
			friend Panel;
		protected:
			// for composition
			Vector2		            m_position;
			Vector2		            m_size;
			Vector2		            m_maxSize;
			Vector4					m_margin;
			Vector4					m_padding;
			int						m_zIndex;
			bool					m_visiable;
			bool					m_wAuto; // signal when width is set to auto
			bool					m_hAuto; // signal when height is set to auto
			float		            m_opticity;
			unsigned	            m_dirtyFlags;
			HorizentalAlignmentEnum m_hAlign;
			VerticalAlignmentEnum	m_vAlign;
			D2D1_MATRIX_3X2_F		m_transform; // relative transform to it's parent
			HUDElement*				m_parent;
		};

		class I2dDeviceResource abstract
		{
		public:
			// Factory resources is actually device free
			virtual void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) = 0;
			virtual void ReleaseDeviceResources() = 0;
		};

		class Panel : public HUDElement
		{
		public:
			~Panel() override;

			typedef std::vector<HUDElement*> ElementListType;

			const ElementListType& Children() const { return m_children; }

			virtual void AddChild(HUDElement* elem);

			virtual void Render(ID2D1DeviceContext* context);
		protected:
			ElementListType m_children;
		};

		// Root element of hud tree, nothing but to provide the logical size of the window
		class HUDCanvas : public Panel
		{
		public:
			HUDCanvas();
			~HUDCanvas() override;
			virtual void UpdateLayout();
			virtual void Render(ID2D1DeviceContext* context);

			void		 SetTarget(RenderableTexture2D* bitmap);

			RenderableTexture2D*
						 GetTarget() const;
			void		 GetViewport(D3D11_VIEWPORT& viewport) const;

			void		 SetChildPosition(int idx, const Vector2& position) { m_children[idx]->SetPosition(position); }

		protected:
			bool					m_clearCanvasWhenRender;
			Color					m_background;
			RenderableTexture2D*	m_targetTex;
		};

		class StackPanel : public Panel
		{

		};

		class ScatterChart : public HUDElement, public I2dDeviceResource
		{
		public:
			ScatterChart(size_t n, float* xItr, float* yItr);

			enum LineInterpolationTypeEnum
			{
				Point,
				Linear,
				Cubic,
			};

			enum PointLabelTypeEnum
			{
				NoPoint,
				Circle,
			};

			void SetDataPointLabel();
			void SetLineInterpolationType(LineInterpolationTypeEnum type);
			void SetXRange(float min, float max);
			void SetYRange(float min, float max);
			void SetFrameGridType();
			void SetChartTitle(const std::wstring& title);
			void SetSequenceColor(const Color& color);
			void SetLineWidth(float width);
			void SetBackground(const Color& color);
			void SetForeground(const Color& color);

			void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) override;
			void ReleaseDeviceResources() override;
		private:

			std::vector<Vector2, AlignedAllocator<XMVECTOR>> m_rawData;
		};

		class MatrixBitmap : public HUDElement, public I2dDeviceResource
		{

		};

		class TextBlock : public HUDElement, public I2dDeviceResource
		{
			TextBlock();
			TextBlock(const std::wstring& text, ID2D1Factory* pD2dFactory = nullptr, IDWriteFactory* pDwriteFactory = nullptr);

			~TextBlock();

			void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) override;
			void ReleaseDeviceResources() override;

			const std::wstring& Text() const;
			void SetText(const std::wstring& text);

			const Color& Foreground() const;
			void SetForeground(const Color& color);

			float FontSize() const;
			void  SetFontSize(float fontSize);

			virtual void Render(ID2D1DeviceContext* context) override;

		private:
			void UpdateBrushColor(ID2D1DeviceContext* pContext);
			// Create New text layout when text is changed
			void UpdateTextLayout(IDWriteFactory* pFactory = nullptr);

		private:
			std::wstring                    m_text;
			Color							m_textColor;
			float							m_fontSize;

			DWRITE_TEXT_METRICS	            m_textMetrics;

			ComPtr<IDWriteFactory>			m_dwFactory;
			ComPtr<ID2D1SolidColorBrush>    m_Brush;
			ComPtr<ID2D1DrawingStateBlock>  m_stateBlock;
			ComPtr<IDWriteTextLayout>       m_textLayout;
			ComPtr<IDWriteTextFormat>		m_textFormat;
		};
	}
}