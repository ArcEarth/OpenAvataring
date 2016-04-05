#include "pch_directX.h"
#include <codecvt>
#include "HUD.h"
#include "DirectXHelper.h"

using namespace DirectX;
using namespace DirectX::Scene;
using namespace D2D1;

wchar_t g_defaultFontFamilayName[] = L"Segoe UI";
wchar_t g_defaultLocaleName[] = L"en-US";
float	g_defaultFontSize = 20.0f;

namespace DirtyFlags
{
	unsigned Position = 0x1;
	unsigned Size = 0x2;
	unsigned Alignment = 0x4;
	unsigned Opticity = 0x8;
	unsigned TextColor = 0x10;
	unsigned TextContent = 0x20;
	unsigned FontSize = 0x40;
}

// Updates the text to be displayed.
//void HUDInterface::UpdateAnimation(DirectX::StepTimer const& timer)
//{
//	m_text = (fps > 0) ? std::to_wstring(fps) + L" FPS" : L" - FPS";
//}

TextBlock::TextBlock()
{
	m_textColor = DirectX::Colors::White.v;
	m_fontSize = g_defaultFontSize;
	ZeroMemory(&m_textMetrics, sizeof(m_textMetrics));
}

TextBlock::TextBlock(const std::wstring & text, ID2D1Factory * pD2dFactory, IDWriteFactory * pDwriteFactory)
{
	m_textColor = DirectX::Colors::White.v;
	m_fontSize = g_defaultFontSize;
	m_text = text;
}

void TextBlock::CreateDeviceResources(ID2D1Factory* pD2dFactory, IDWriteFactory * pFactory)
{
	DirectX::ThrowIfFailed(
		pFactory->CreateTextFormat(
			g_defaultFontFamilayName,
			nullptr,
			DWRITE_FONT_WEIGHT_LIGHT,
			DWRITE_FONT_STYLE_NORMAL,
			DWRITE_FONT_STRETCH_NORMAL,
			m_fontSize,
			g_defaultLocaleName,
			&m_textFormat
			)
		);

	ThrowIfFailed(
		m_textFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_TRAILING)
		);

	DirectX::ThrowIfFailed(
		m_textFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR)
		);

	if (!m_text.empty())
		UpdateTextLayout(pFactory);
}

void TextBlock::ReleaseDeviceResources()
{
	m_Brush.Reset();
	m_stateBlock.Reset();
	m_textLayout.Reset();
	m_textFormat.Reset();
}

const std::wstring & TextBlock::Text() const { return m_text; }

void TextBlock::SetText(const std::wstring & text) {
	m_text = text; 
	m_dirtyFlags |= DirtyFlags::TextContent; 
}

void TextBlock::SetText(const std::string & text_utf8)
{
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	this->SetText(converter.from_bytes(text_utf8));
}

const Color & TextBlock::Foreground() const { return m_textColor; }

void TextBlock::SetForeground(const Color & color) { m_textColor = color; m_dirtyFlags |= DirtyFlags::TextColor; }

float TextBlock::FontSize() const { return m_fontSize; }

void TextBlock::SetFontSize(float fontSize) { m_fontSize = fontSize; m_dirtyFlags |= DirtyFlags::FontSize; }

void TextBlock::UpdateBrushColor(ID2D1DeviceContext* pContext)
{
	const auto& color =
		reinterpret_cast<const D2D1_COLOR_F &>(m_textColor);
	if (m_Brush)
	{
		if (m_dirtyFlags & DirtyFlags::TextColor)
		{
			m_dirtyFlags &= ~DirtyFlags::TextColor;
			m_Brush->SetColor(color);
		}

		if (m_dirtyFlags & DirtyFlags::Opticity)
		{
			m_dirtyFlags &= ~DirtyFlags::Opticity;
			m_Brush->SetOpacity(m_opticity);
		}
	}
	else
	{
		m_dirtyFlags &= ~DirtyFlags::TextColor;
		m_dirtyFlags &= ~DirtyFlags::Opticity;
		pContext->CreateSolidColorBrush(color, &m_Brush);
		m_Brush->SetOpacity(m_opticity);
	}
}

void TextBlock::UpdateTextLayout(IDWriteFactory* pFactory)
{
	if (pFactory == nullptr)
		pFactory = m_dwFactory.Get();

	if (pFactory == nullptr)
		return;

	if (m_textLayout == nullptr || m_dirtyFlags & DirtyFlags::TextContent)
	{
		m_dirtyFlags &= ~DirtyFlags::TextContent;
		DirectX::ThrowIfFailed(
			pFactory->CreateTextLayout(
				m_text.c_str(),
				(uint32_t)m_text.length(),
				m_textFormat.Get(),
				m_size.x, // Max width of the input text.
				m_size.y, // Max height of the input text.
				&m_textLayout
				)
			);
	}
	else if (m_dirtyFlags & DirtyFlags::Size)
	{
		m_dirtyFlags &= ~DirtyFlags::Size;
		m_textLayout->SetMaxHeight(m_size.y);
		m_textLayout->SetMaxWidth(m_size.x);
	}
	else if (m_dirtyFlags & DirtyFlags::FontSize)
	{
		m_dirtyFlags &= ~DirtyFlags::FontSize;
		m_textLayout->SetFontSize(m_fontSize, DWRITE_TEXT_RANGE{ 0U, (UINT32)m_text.size() });
	}

	DirectX::ThrowIfFailed(
		m_textLayout->GetMetrics(&m_textMetrics)
		);
}


void TextBlock::Render(ID2D1DeviceContext* context)
{
	if (m_text.empty() || !m_visiable || m_opticity == .0f) return;

	UpdateLayout();
	UpdateTextLayout();
	UpdateBrushColor(context);

	//if (!m_textLayout)
	//	throw std::runtime_error("not intialized");

	auto state = SetTransform(context);

	context->DrawTextLayout(
		D2D1::Point2F(0.f, 0.f),
		m_textLayout.Get(),
		m_Brush.Get()
		);

	// Ignore D2DERR_RECREATE_TARGET here. This error indicates that the device
	// is lost. It will be handled during the next call to Present.

	RestoreTransform(context,state);
}

TextBlock::~TextBlock()
{}

Vector2 HUDElement::MeasureOverride(const Vector2 & availableSize)
{
	return m_size;
}

Vector2 HUDElement::ArrangeOverride(const Vector2 & actualSize)
{
	return m_size;
}

void HUDElement::Render(ID2D1DeviceContext * context)
{
}

bool HUDElement::IsDirety() const { return m_dirtyFlags != 0; }

void HUDElement::CreateDeviceResources(ID2D1Factory * pD2dFactory, IDWriteFactory * pDwriteFactory)
{
}

void HUDElement::ReleaseDeviceResources()
{
}

HUDElement::HUDElement()
	: m_parent(nullptr),
	m_transform(D2D1::Matrix3x2F::Identity()),
	m_hAlign(HorizentalAlignmentEnum::Left), m_vAlign(VerticalAlignmentEnum::Top),
	m_dirtyFlags(DirtyFlags::Size | DirtyFlags::Alignment | DirtyFlags::Position),
	m_opticity(1.0f),
	m_wAuto(true), m_hAuto(true),
	m_position(), m_size(),
	m_visiable(true)
{
}

HUDElement::~HUDElement()
{}


float HUDElement::Width() const { 
	return Size().x;
}

float HUDElement::Height() const { 
	return Size().y;
}

void HUDElement::SetWidth(float width) {
	m_size.x = width;
	m_dirtyFlags |= DirtyFlags::Size;
}

void HUDElement::SetHeight(float height) {
	m_size.y = height;
	m_dirtyFlags |= DirtyFlags::Size;
}

Vector2 HUDElement::Size() const {
	if (m_dirtyFlags & DirtyFlags::Size)
		const_cast<HUDElement*>(this)->UpdateLayout();

	return m_size;
}

Vector2 HUDElement::Position() const {
	if (m_dirtyFlags & DirtyFlags::Position)
		const_cast<HUDElement*>(this)->UpdateLayout();

	return m_position;
}

void HUDElement::SetSize(const Vector2 & size) {
	m_size = size;
	m_dirtyFlags |= DirtyFlags::Size;
}

void HUDElement::SetPosition(const Vector2 & position) {
	m_position = position;
	m_dirtyFlags |= DirtyFlags::Position;
}

void HUDElement::UpdateLayout()
{
	if (IsLayoutValiad())
		return;

	auto logicalSize = Parent()->Size();

	if (m_hAlign == HorizentalAlignmentEnum::Stretch)
		m_size.x = logicalSize.x;
	if (m_vAlign == VerticalAlignmentEnum::Stretch)
		m_size.y = logicalSize.y;

	float tx = 0, ty = 0, sx = 1, sy = 1;

	switch (m_hAlign)
	{
	case HorizentalAlignmentEnum::Right:
		tx = logicalSize.x - m_size.x;
		break;
	case HorizentalAlignmentEnum::Center:
		tx = (logicalSize.x - m_size.x) * 0.5f;
		break;
	case HorizentalAlignmentEnum::Stretch:
	case HorizentalAlignmentEnum::Left:
	default:
		break;
	}

	switch (m_vAlign)
	{
	case VerticalAlignmentEnum::Bottom:
		tx = logicalSize.y - m_size.y;
		break;
	case VerticalAlignmentEnum::Center:
		tx = (logicalSize.y - m_size.y) * 0.5f;
		break;
	case VerticalAlignmentEnum::Stretch:
	case VerticalAlignmentEnum::Top:
	default:
		break;
	}

	m_transform = Matrix3x2F::Translation(
		tx,
		ty
		);

	m_dirtyFlags &= ~(DirtyFlags::Size | DirtyFlags::Alignment | DirtyFlags::Position);
}

ID2D1DrawingStateBlock* HUDElement::SaveDrawingState(ID2D1DeviceContext * context)
{
	if (m_stateBlock == nullptr)
	{
		ComPtr<ID2D1Factory> pD2dFactory;
		context->GetFactory(&pD2dFactory);
		DirectX::ThrowIfFailed(
			pD2dFactory->CreateDrawingStateBlock(&m_stateBlock)
			);
	}

	context->SaveDrawingState(m_stateBlock.Get());

	return m_stateBlock.Get();
}

D2D1_MATRIX_3X2_F HUDElement::SetTransform(ID2D1DeviceContext * context)
{
	Matrix3x2F parentTransform;
	context->GetTransform(&parentTransform);
	context->SetTransform(m_transform * parentTransform);
	return parentTransform;
}

void HUDElement::RestoreTransform(ID2D1DeviceContext * context, const D2D1_MATRIX_3X2_F &parentTransform)
{
	context->SetTransform(parentTransform);
}

void HUDElement::RestoreDrawingState(ID2D1DeviceContext * context)
{
	if (m_stateBlock)
		context->RestoreDrawingState(m_stateBlock.Get());
}

void HUDElement::SetOpticity(float opticity) {
	m_opticity = opticity;
	m_dirtyFlags |= DirtyFlags::Opticity;
}

void HUDElement::SetHorizentalAlignment(HorizentalAlignmentEnum align) {
	m_dirtyFlags |= m_hAlign == align ? 0 : DirtyFlags::Alignment;
	m_hAlign = align;
}

void HUDElement::SetVerticalAlignment(VerticalAlignmentEnum align) {
	m_dirtyFlags |= m_vAlign == align ? 0 : DirtyFlags::Alignment;
	m_vAlign = align;
}

bool HUDElement::IsLayoutValiad() const {
	return !(m_dirtyFlags & (DirtyFlags::Position | DirtyFlags::Alignment | DirtyFlags::Size))
		&& (!m_parent || m_parent->IsLayoutValiad());
}

void HUDElement::InvaliadLayout()
{
	m_dirtyFlags |= DirtyFlags::Size | DirtyFlags::Alignment | DirtyFlags::Position;
}

Vector2 HUDElement::Measure(const Vector2 & availableSize)
{
	return MeasureOverride(availableSize);
}

Vector2 HUDElement::Arrange(const Vector2 & actualSize)
{
	return ArrangeOverride(actualSize);
}

void HUDCanvas::UpdateLayout()
{
	m_position.x = .0f;
	m_position.y = .0f;
	m_size.x = m_targetTex ? m_targetTex->Width() : .0f;
	m_size.y = m_targetTex ? m_targetTex->Height() : .0f;
	m_dirtyFlags &= ~(DirtyFlags::Size | DirtyFlags::Alignment | DirtyFlags::Position);
}

HUDCanvas::~HUDCanvas()
{
}

void HUDCanvas::Render(ID2D1DeviceContext * context)
{
	if (!m_targetTex)
		return;

	context->SetTarget(m_targetTex->BitmapView());
	if (m_clearCanvasWhenRender)
		context->Clear(as_d2d(m_background));

	UpdateLayout();

	if (m_children.empty())
		return;

	context->SetTransform(D2D1::Matrix3x2F::Identity());
	context->BeginDraw();
	Panel::Render(context);
	context->EndDraw();
}

void HUDCanvas::SetBackground(Color color)
{
	m_background = color;
	m_clearCanvasWhenRender = true;
}

std::unique_ptr<RenderableTexture2D> DirectX::Scene::HUDCanvas::CreateTarget(ID3D11Device * pDevice, ID2D1DeviceContext * pD2DContext, size_t width, size_t height)
{
	std::unique_ptr<RenderableTexture2D> canvas;
	canvas.reset(new RenderableTexture2D(pDevice, width, height, DXGI_FORMAT_B8G8R8A8_UNORM, 1, 0, true));
	canvas->CreateD2DBitmapView(pD2DContext);
	return canvas;
}

HUDCanvas::HUDCanvas()
	: m_targetTex(nullptr), m_background(Colors::Transparent.v), m_clearCanvasWhenRender(false)
{}

void HUDCanvas::SetTarget(RenderableTexture2D* rtex)
{
	m_targetTex = rtex;
	auto bitmapView = rtex->BitmapView();
	assert(bitmapView != nullptr && "The given surface does not support D2D interface");

	auto size = bitmapView->GetSize();
	m_size.x = size.width;
	m_size.y = size.height;
	m_position.x = 0;
	m_position.y = 0;
}

RenderableTexture2D* HUDCanvas::GetTarget() const
{
	return m_targetTex;
}

void HUDCanvas::GetViewport(D3D11_VIEWPORT& viewport) const
{
	viewport.TopLeftX = m_position.x;
	viewport.TopLeftY = m_position.y;
	viewport.Width    = m_size.x;
	viewport.Height   = m_size.y;
}

void Panel::AddChild(const std::shared_ptr<HUDElement>& elem)
{
	m_children.push_back(elem);
	elem->m_parent = this;
}

void Panel::Render(ID2D1DeviceContext * context)
{
	auto state = SetTransform(context);

	for (auto& element : m_children)
	{
		element->Render(context);
	}

	RestoreTransform(context, state);
}

Panel::~Panel()
{
}

ProgressBar::ProgressBar(int width, int height, TextIndicationOption textOpt, ElementOrientation orientation)
{
	m_size = Vector2(width,height);
	m_orientaion = orientation;
}

void ProgressBar::Render(ID2D1DeviceContext * context)
{
	UpdateLayout();
	auto state = SetTransform(context);
	D2D1_RECT_F rect;
	rect.left = rect.right = .0f;
	rect.bottom = m_size.y;
	rect.right = m_size.x;
	context->FillRectangle(rect, m_backgroundBrush.Get());

	if (m_orientaion == ElementOrientation::Horizental)
		rect.right = (m_value - m_min) / (m_max - m_min) * m_size.x;
	else
		rect.bottom = (m_value - m_min) / (m_max - m_min) *m_size.y;

	context->FillRectangle(rect, m_foregroundBrush.Get());
	RestoreTransform(context, state);
}
