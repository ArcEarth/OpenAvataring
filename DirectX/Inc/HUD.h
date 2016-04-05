#pragma once

#include <string>
#include <memory>
#include <limits>
#include <dwrite_2.h>
#include <d2d1_2.h>
#include "DirectXMathExtend.h"
#include "DirectXMathSimpleVectors.h"
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

	inline DirectX::XMVECTOR XM_CALLCONV XMParticleProjection(DirectX::FXMVECTOR V, DirectX::FXMMATRIX worldView, DirectX::CXMMATRIX proj, const D3D11_VIEWPORT& vp)
	{
		using namespace DirectX;
		const float HalfViewportWidth = vp.Width * 0.5f;
		const float HalfViewportHeight = vp.Height * 0.5f;

		XMVECTOR Scale = XMVectorSet(HalfViewportWidth, -HalfViewportHeight, vp.MaxDepth - vp.MinDepth, 0.0f);
		XMVECTOR Offset = XMVectorSet(vp.TopLeftY + HalfViewportWidth, vp.TopLeftY + HalfViewportHeight, vp.MinDepth, 0.0f);

		XMVECTOR R = _DXMEXT XMVectorSplatW(V); // particle radius
		XMVECTOR P = _DXMEXT XMVectorSetW(V, 1.0f);
		P = _DXMEXT XMVector3Transform(V, worldView);

		R = _DXMEXT XMVectorPermute<0, 1, 6, 7>(R, P); // (r,r,depth,1.0)

		P = _DXMEXT XMVector3TransformCoord(P, proj);
		R = _DXMEXT XMVector3TransformCoord(R, proj);

		P = _DXMEXT XMVectorMultiplyAdd(P, Scale, Offset);
		R = XMVectorMultiply(R, Scale);

		P = _DXMEXT XMVectorPermute<0, 1, 2, 4>(P, R); // (X,Y,Z,size)
		return P;
	}

	inline const D2D1_COLOR_F& as_d2d(const DirectX::Color& color)
	{
		return reinterpret_cast<const D2D1_COLOR_F&>(color);
	}

	namespace Scene
	{
		class TextBlock;

		using Microsoft::WRL::ComPtr;

		enum class ElementOrientation
		{
			Horizental,
			Vertical,
		};

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

		constexpr float AutoSize = std::numeric_limits<float>::signaling_NaN();

		class I2dDeviceResource abstract
		{
		public:
			// Factory resources is actually device free
			virtual void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) = 0;
			virtual void ReleaseDeviceResources() = 0;
		};

		XM_ALIGNATTR
		class HUDElement : public AlignedNew<XMVECTOR>, public I2dDeviceResource
		{
		public:
			HUDElement();
			virtual ~HUDElement();

			float Width() const;
			float Height() const;
			void  SetWidth(float width);
			void  SetHeight(float height);

			Vector2 Size() const;
			Vector2 Position() const;

			void SetSize(const Vector2& size);
			void SetPosition(const Vector2& position);

			float Opticity() const { return m_opticity; }
			void SetOpticity(float opticity);

			bool Visiability() const { return m_visiable; }
			void SetVisiability(bool visiable) { m_visiable = visiable; }

			int	ZIndex() const { return m_zIndex; }
			void SetZIndex(int zIndex) { m_zIndex = zIndex; }

			HorizentalAlignmentEnum HorizentalAlignment() const { return m_hAlign; };
			VerticalAlignmentEnum	VerticalAlignment() const { return m_vAlign; };

			void SetHorizentalAlignment(HorizentalAlignmentEnum align);;
			void SetVerticalAlignment(VerticalAlignmentEnum align);;
			
			virtual HUDElement* HitTest(Vector2 point) {
				Vector2 lt = Position();
				Vector2 rb = lt + Size();
				return 
					XMVector2LessOrEqual(lt, point)
					&& XMVector2Less(point, rb) ?
					this : nullptr;
			}

			const HUDElement* HitTest(Vector2 point) const
			{
				return const_cast<HUDElement*>(this)->HitTest(point);
			}

			const HUDElement* Parent() const { return m_parent; }
			HUDElement* Parent() { return m_parent; }

			// Layout interface
			bool IsLayoutValiad() const;
			void InvaliadLayout();

			Vector2 Measure(const Vector2& availableSize);
			Vector2 Arrange(const Vector2& actualSize);
			virtual Vector2 MeasureOverride(const Vector2& availableSize);
			virtual Vector2 ArrangeOverride(const Vector2& actualSize);
			virtual void UpdateLayout();

			ID2D1DrawingStateBlock*
				SaveDrawingState(ID2D1DeviceContext* context);
			void RestoreDrawingState(ID2D1DeviceContext* context);

			D2D1_MATRIX_3X2_F SetTransform(ID2D1DeviceContext* context);
			void RestoreTransform(ID2D1DeviceContext * context, const D2D1_MATRIX_3X2_F &parentTransform);

			// Rendering interface
			virtual void Render(ID2D1DeviceContext* context);

			virtual bool IsDirety() const;


			void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) override;
			void ReleaseDeviceResources() override;

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
			ComPtr<ID2D1DrawingStateBlock>  
									m_stateBlock;
			HUDElement*				m_parent;
		};

		class Panel : public HUDElement
		{
		public:
			~Panel() override;

			typedef std::shared_ptr<HUDElement> ElementPtr;
			typedef std::vector<std::shared_ptr<HUDElement>> ElementListType;

			const ElementListType& Children() const { return m_children; }

			void AddChild(const std::shared_ptr<HUDElement>& elem);

			void Render(ID2D1DeviceContext* context) override;

			virtual bool IsDirety() const override {
				return std::any_of(m_children.begin(), m_children.end(),
					[](const ElementPtr& child) -> bool {
					return child->IsDirety();
				});
			}

			void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) override
			{
				for (auto& child : m_children)
				{
					child->CreateDeviceResources(pD2dFactory, pDwriteFactory);
				}
			}
			void ReleaseDeviceResources() override
			{
				for (auto& child : m_children)
				{
					child->ReleaseDeviceResources();
				}
			}

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
			void Render(ID2D1DeviceContext* context) override;

			void SetBackground(Color color);

			Color Background() const { return m_background; }

			static
			std::unique_ptr<RenderableTexture2D>
						 CreateTarget(ID3D11Device* pDevice, ID2D1DeviceContext* pD2DContext, size_t width, size_t height);

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
		public:

		protected:
			ElementOrientation m_orientation;
		};

		class ScatterChart : public HUDElement
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

		class MatrixBitmap : public HUDElement
		{

		};

		class TextBlock : public HUDElement
		{
		public:
			TextBlock();
			TextBlock(const std::wstring& text, ID2D1Factory* pD2dFactory = nullptr, IDWriteFactory* pDwriteFactory = nullptr);

			~TextBlock();

			void CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory* pDwriteFactory) override;
			void ReleaseDeviceResources() override;

			const std::wstring& Text() const;
			void SetText(const std::wstring& text);
			void SetText(const std::string& text_utf8);

			const Color& Foreground() const;
			void SetForeground(const Color& color);

			float FontSize() const;
			void  SetFontSize(float fontSize);

			virtual void Render(ID2D1DeviceContext* context) override;

		private:
			void UpdateBrushColor(ID2D1DeviceContext* pContext);
			// Create New text layout when text is changed
			void UpdateTextLayout(IDWriteFactory* pFactory = nullptr);

		protected:
			std::wstring                    m_text;
			Color							m_textColor;
			float							m_fontSize;

			DWRITE_TEXT_METRICS	            m_textMetrics;

			ComPtr<IDWriteFactory>			m_dwFactory;
			ComPtr<ID2D1SolidColorBrush>    m_Brush;
			ComPtr<IDWriteTextLayout>       m_textLayout;
			ComPtr<IDWriteTextFormat>		m_textFormat;
		};

		class ProgressBar : public HUDElement
		{
		public:
			enum TextIndicationOption
			{
				TextIndication_None,
				TextIndication_Percentage,
				TextIndication_ValueSlashMaxMinusMin,
			};

			ProgressBar(int width = AutoSize, int height = AutoSize, TextIndicationOption textOpt = TextIndication_None, ElementOrientation orientation = ElementOrientation::Horizental);

			double Value() const { return m_value; }
			void SetValue(double val) { m_value = val; }

			double Min() const { return m_min; }
			void SetMin(double val) { m_min = val; }

			double Max() const { return m_max; }
			void SetMax(double val) { m_max = val; }

			void Render(ID2D1DeviceContext* context) override;

			ElementOrientation m_orientaion;

			bool   m_showValueText;

			double m_max, m_min, m_value;

			Color  m_foreground;
			Color  m_background;

			ComPtr<ID2D1SolidColorBrush>    m_foregroundBrush;
			ComPtr<ID2D1SolidColorBrush>    m_backgroundBrush;

			TextIndicationOption			m_textOption;
			std::shared_ptr<TextBlock>		m_textBlock;
		};
	}
}