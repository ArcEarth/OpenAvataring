#include "pch_directX.h"
#include "PostProcessingEffect.h"
#include "SharedResourcePool.h"
#include "DemandCreate.h"
#include "ConstantBuffer.h"
#include "Models.h"
#include <VertexTypes.h>
#include <CommonStates.h>

using namespace DirectX;
using namespace DirectX::HLSLVectors;
// Points to a precompiled vertex or pixel shader program.
struct ShaderBytecode
{
	void const* code;
	size_t length;
};

template <size_t Size>
inline ShaderBytecode MakeShaderByteCode(const BYTE(&bytecode)[Size])
{
	return ShaderBytecode{ bytecode ,sizeof(bytecode) };
}


using Microsoft::WRL::ComPtr;

struct QuadVertexShaderConstant
{
	float2 uv_base;
	float2 uv_range;
};

namespace DirectX
{
	namespace EffectDirtyFlags
	{
		static const int VSConstantBuffer = 0x400;
		static const int PSConstantBuffer = 0x800;
	}
}

// Factory for lazily instantiating shaders. BasicEffect supports many different
// shader permutations, so we only bother creating the ones that are actually used.
class PostEffectDeviceResources
{
private:
	ID3D11VertexShader* DemandCreateVertexShader(_Inout_ Microsoft::WRL::ComPtr<ID3D11VertexShader>& vertexShader, ShaderBytecode const& bytecode);

public:
	PostEffectDeviceResources(_In_ ID3D11Device* device)
		: m_Device(device), m_VSCBuffer(device)
	{
		GetVertexShader();
	}

	// Gets or lazily creates the specified vertex shader permutation.
	ID3D11VertexShader* GetVertexShader()
	{
		return DemandCreateVertexShader(m_QuadVertexShader, s_QuadVertexShaderBtyeCode);
	}

	ConstantBuffer<QuadVertexShaderConstant>& GetVSCbuffer() { return m_VSCBuffer; }

	ID3D11PixelShader * DemandCreatePixelShader(_Inout_ Microsoft::WRL::ComPtr<ID3D11PixelShader> & pixelShader, ShaderBytecode const& bytecode);

	void DrawQuad(ID3D11DeviceContext* pContext)
	{
		assert(!m_QuadMesh.Empty());
		m_QuadMesh.Draw(pContext);
	}

	static const ShaderBytecode s_QuadVertexShaderBtyeCode;
protected:
	Microsoft::WRL::ComPtr<ID3D11Device>			 m_Device;
	Microsoft::WRL::ComPtr<ID3D11VertexShader>		 m_QuadVertexShader;
	Scene::MeshBuffer								 m_QuadMesh;
	ConstantBuffer<QuadVertexShaderConstant>		 m_VSCBuffer;

	std::mutex										 m_Mutex;

};

static VertexPositionTexture QuadVertices[] = {
	{{ -1.0f,  1.0f, 0.5f, 1.0f }, { 0.0f, 0.0f }},
	{{ 1.0f ,  1.0f, 0.5f, 1.0f }, { 1.0f, 0.0f }},
	{{ -1.0f, -1.0f, 0.5f, 1.0f }, { 0.0f, 1.0f }},
	{{ 1.0f , -1.0f, 0.5f, 1.0f }, { 1.0f, 1.0f }}
};

// Gets or lazily creates the specified vertex shader permutation.
ID3D11VertexShader* PostEffectDeviceResources::DemandCreateVertexShader(_Inout_ ComPtr<ID3D11VertexShader>& vertexShader, ShaderBytecode const& bytecode)
{
	return DemandCreate(vertexShader, m_Mutex, [&](ID3D11VertexShader** pResult) -> HRESULT
	{
		HRESULT hr = m_Device->CreateVertexShader(bytecode.code, bytecode.length, nullptr, pResult);

		if (SUCCEEDED(hr))
			SetDebugObjectName(*pResult, "DirectXTK:PostEffectQuadVertexShader");

		m_QuadMesh.CreateDeviceResources<VertexPositionTexture>(m_Device.Get(), QuadVertices, std::size(QuadVertices), nullptr, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
		m_QuadMesh.CreateInputLayout(m_Device.Get(), bytecode.code, bytecode.length);
		return hr;
	});
}


// Gets or lazily creates the specified pixel shader permutation.
ID3D11PixelShader* PostEffectDeviceResources::DemandCreatePixelShader(_Inout_ ComPtr<ID3D11PixelShader>& pixelShader, ShaderBytecode const& bytecode)
{
	return DemandCreate(pixelShader, m_Mutex, [&](ID3D11PixelShader** pResult) -> HRESULT
	{
		HRESULT hr = m_Device->CreatePixelShader(bytecode.code, bytecode.length, nullptr, pResult);

		if (SUCCEEDED(hr))
			SetDebugObjectName(*pResult, "DirectXTK:PostEffectPixelShaders");

		return hr;
	});
}

using Microsoft::WRL::ComPtr;
// Templated base class provides functionality common to all the built-in effects.
template<typename Traits>
class PostProcessingEffectBase : public AlignedNew<XMVECTOR>
{
public:
	// Constructor.
	PostProcessingEffectBase(_In_ ID3D11Device* device)
		: dirtyFlags(INT_MAX),
		m_ConstantBuffer(device),
		m_DeviceResources(s_DeviceResourcesPool.DemandCreate(device)),
		states(device)
		{
			ZeroMemory(&constants, sizeof(constants));
			SetInputViewport(float2(0, 0), float2(1, 1));
		}

	std::vector<ID3D11ShaderResourceView*>	inputs;
	typename Traits::ConstantBufferType		constants;
	CommonStates							states;
	int dirtyFlags;
	// Helper sets our shaders and constant buffers onto the D3D device.
	void RenderPass(_In_ ID3D11DeviceContext* deviceContext, int pass)
	{
		// Set shaders.
		auto vertexShader = m_DeviceResources->GetVertexShader();
		auto pixelShader = m_DeviceResources->GetPixelShader(pass);

		deviceContext->VSSetShader(vertexShader, nullptr, 0);
		deviceContext->PSSetShader(pixelShader, nullptr, 0);

		auto& vsCbuffer = m_DeviceResources->GetVSCbuffer();
		// Make sure the constant buffer is up to date.
		if (dirtyFlags & EffectDirtyFlags::VSConstantBuffer)
		{
			vsCbuffer.SetData(deviceContext, inputViewport);

			dirtyFlags &= ~EffectDirtyFlags::VSConstantBuffer;
		}
		// Make sure the constant buffer is up to date.
		if (dirtyFlags & EffectDirtyFlags::PSConstantBuffer)
		{
			m_ConstantBuffer.SetData(deviceContext, constants);

			dirtyFlags &= ~EffectDirtyFlags::PSConstantBuffer;
		}
		// Set the constant buffer.
		ID3D11Buffer* psbuffer = m_ConstantBuffer.GetBuffer();
		ID3D11Buffer* vsbuffer = vsCbuffer.GetBuffer();
		deviceContext->VSSetConstantBuffers(0, 1, &vsbuffer);
		deviceContext->PSSetConstantBuffers(0, 1, &psbuffer);
		m_DeviceResources->DrawQuad(deviceContext);
	}

	void SetInputResources(UINT numSource, ID3D11ShaderResourceView * const * pSources)
	{
		inputs.clear();
		std::copy_n(pSources, numSource, std::back_inserter(inputs));
	}

	void SetInputViewport(const D3D11_VIEWPORT &viewport,const XMUINT2& textureSize)
	{
		inputViewport.uv_base = float2{ viewport.TopLeftX / (float)textureSize.x,viewport.TopLeftY / (float)textureSize.y };
		inputViewport.uv_range = float2{ viewport.Width / (float)textureSize.x,viewport.Height / (float)textureSize.y };
		dirtyFlags |= EffectDirtyFlags::VSConstantBuffer;
	}

	void SetInputViewport(const float2 &leftTop, const float2 &size)
	{
		inputViewport.uv_base = leftTop;
		inputViewport.uv_range = size;
		dirtyFlags |= EffectDirtyFlags::VSConstantBuffer;
	}

	float2 GetInputViewportSize() const
	{
		return inputViewport.uv_range;
	}

	float2 GetInputViewportBase() const
	{
		return inputViewport.uv_base;
	}


protected:
	// Static arrays hold all the precompiled shader permutations.
	static const ShaderBytecode PixelShaderBytecode[Traits::PixelShaderCount];

	static const int PixelShaderIndices[Traits::PassCount];


private:
	// Fields.
	QuadVertexShaderConstant			inputViewport;

	// D3D constant buffer holds a copy of the same data as the public 'constants' field.
	ConstantBuffer<typename Traits::ConstantBufferType> m_ConstantBuffer;

	// Only one of these helpers is allocated per D3D device, even if there are multiple effect instances.
	class DeviceResources : public PostEffectDeviceResources
	{
	public:
		DeviceResources(_In_ ID3D11Device* device)
			: PostEffectDeviceResources(device)
		{ }

		// Gets or lazily creates the specified pixel shader permutation.
		ID3D11PixelShader* GetPixelShader(int pass)
		{
			int shaderIndex = PixelShaderIndices[pass];

			return DemandCreatePixelShader(m_PixelShaders[shaderIndex], PixelShaderBytecode[shaderIndex]);
		}

	private:
		Microsoft::WRL::ComPtr<ID3D11PixelShader>  m_PixelShaders[Traits::PixelShaderCount];
	};

	// Per-device resources.
	std::shared_ptr<DeviceResources> m_DeviceResources;

	static SharedResourcePool<ID3D11Device*, DeviceResources> s_DeviceResourcesPool;
};

namespace
{
	static const uint MaxSampleCount = 15;
	XM_ALIGNATTR
	struct BlurCBufferType : public AlignedNew<XMVECTOR>
	{
		float4 sampleOffsets[MaxSampleCount];
		float4 sampleWeights[MaxSampleCount];
		uint   sampleCount;
		float pixelWidth;
		float pixelHeight;
		float muiltiplier;
	};
	static_assert(sizeof(BlurCBufferType) == 496,"Lalla");
}
struct BlurEffectTraits
{
	typedef BlurCBufferType ConstantBufferType;
	static const int PixelShaderCount = 5;
	static const int PassCount = 6;
};

typedef PostProcessingEffectBase<BlurEffectTraits> BlurEffectBase;

namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
#include "Shaders/Xbox/QuadVertexShader_QuadVertexShader.inc"

#include "Shaders/Xbox/BlurEffect_DownScale3x3Ex.inc"
#include "Shaders/Xbox/BlurEffect_Blur.inc"
#else
#include "Shaders/Windows/QuadVertexShader_QuadVertexShader.inc"

#include "Shaders/Windows/BlurEffect_DownScale3x3Ex.inc"
#include "Shaders/Windows/BlurEffect_Blur.inc"
#include "Shaders/Windows/BlurEffect_Combination.inc"
#include "Shaders/Windows/BlurEffect_AlphaAsDepthPassBy.inc"
#include "Shaders/Windows/BlurEffect_PassBy.inc"
#endif
}

const ShaderBytecode PostEffectDeviceResources::s_QuadVertexShaderBtyeCode = MakeShaderByteCode(QuadVertexShader_QuadVertexShader);

SharedResourcePool<ID3D11Device*, BlurEffectBase::DeviceResources> BlurEffectBase::s_DeviceResourcesPool;

const ShaderBytecode BlurEffectBase::PixelShaderBytecode[] = {
	MakeShaderByteCode(BlurEffect_DownScale3x3Ex),
	MakeShaderByteCode(BlurEffect_Blur),
	MakeShaderByteCode(BlurEffect_Combination),
	MakeShaderByteCode(BlurEffect_AlphaAsDepthPassBy),
	MakeShaderByteCode(BlurEffect_PassBy),
};

const int BlurEffectBase::PixelShaderIndices[] = {
	0, // Downscale
	1, // Horizental blur
	1, // vertical blur

	// Alternate based output mode
	2, // Bloom Combine with original texture
	3, // Alpha as depth pass by
	4, // Alpha blend output
};

// Base class for all 'separtable' blur kernal class
class BlurEffect : public BlurEffectBase
{
public:
	ID3D11DepthStencilView *m_DepthStencil;
	DoubleBufferTexture2D m_SwapBuffer;
	PostEffectOutputMode  m_Mode;
public:
	BlurEffect(ID3D11Device * pDevice)
		: BlurEffectBase(pDevice), m_Mode(BloomCombination), m_DepthStencil(NULL)
	{
		constants.muiltiplier = 1.0f;
	}

	virtual void SetupSampleOffsetsWeights(bool horizental) = 0;

	void ResizeBufferRespectTo(ID3D11Device * pDevice, const Texture2D & texture)
	{
		ResizeBuffer(pDevice, texture.Bounds(), texture.Format());
	}

	void ResizeBuffer(ID3D11Device * pDevice, const XMUINT2 & size, DXGI_FORMAT format)
	{
		auto& buffer = m_SwapBuffer;
		if (buffer.Width() != size.x || buffer.Height() != size.y || buffer.Format() != format)
			buffer = DoubleBufferTexture2D(pDevice, size.x, size.y, format);
	}

	void SetupPixelSize(float textureWidth, float textureHeight)
	{
		const float epsilon = XM_EPSILON * 10;
		float texelWidth = 1.0f / textureWidth;
		float texelHeight = 1.0f / textureHeight;
		if (abs(constants.pixelHeight - texelHeight) > epsilon || abs(constants.pixelWidth - texelWidth) > epsilon)
		{
			constants.pixelHeight = texelHeight;
			constants.pixelWidth = texelWidth;
			dirtyFlags |= EffectDirtyFlags::PSConstantBuffer;
		}
	};

	void Apply(ID3D11DeviceContext *pContext, RenderableTexture2D& target, const Texture2D& source, const D3D11_VIEWPORT &outputviewport, const D3D11_VIEWPORT & inputViewport)
	{

		ID3D11SamplerState* pSamplers[] = { states.PointClamp(), states.LinearClamp() };
		CD3D11_VIEWPORT bufferviewport(.0f, .0f, (float)m_SwapBuffer.Width(), (float)m_SwapBuffer.Height());

		ID3D11RenderTargetView* pRTVs[1] = { m_SwapBuffer.RenderTargetView() };
		ID3D11ShaderResourceView* pSRVs[2] = { source.ShaderResourceView(),NULL };
		ID3D11ShaderResourceView* pNullSRVs[2] = {NULL,NULL};
		SetInputViewport(inputViewport,source.Bounds());

		// disable depth stencil & setup samplers
		pContext->OMSetDepthStencilState(states.DepthNone(), 0);
		pContext->OMSetBlendState(states.Opaque(), g_XMOne.f, -1);
		pContext->PSSetSamplers(0, 2, pSamplers);
		pContext->RSSetViewports(1, &bufferviewport);

		// Down scale pass
		SetupPixelSize(m_SwapBuffer.Width(), m_SwapBuffer.Height());
		//pContext->ClearRenderTargetView(pRTVs[0], g_XMZero.f);
		pContext->OMSetRenderTargets(1, pRTVs, NULL);
		pContext->PSSetShaderResources(0, 1, pSRVs);
		RenderPass(pContext, 0);

		// Horizental blur
		m_SwapBuffer.SwapBuffer();
		pRTVs[0] = m_SwapBuffer.RenderTargetView();
		pSRVs[0] = m_SwapBuffer.ShaderResourceView();
		SetInputViewport(bufferviewport,m_SwapBuffer.Bounds());
		SetupSampleOffsetsWeights(true);
		//pContext->ClearRenderTargetView(pRTVs[0], g_XMZero.f);

		pContext->PSSetShaderResources(0, 1, pNullSRVs);
		pContext->OMSetRenderTargets(1, pRTVs, NULL);
		pContext->PSSetShaderResources(0, 1, pSRVs);
		RenderPass(pContext, 1);

		// Vertical blur
		m_SwapBuffer.SwapBuffer();
		pSRVs[0] = m_SwapBuffer.ShaderResourceView();
		pRTVs[0] = m_SwapBuffer.RenderTargetView();// target.RenderTargetView(); 		pContext->RSSetViewports(1, &outputviewport);
		SetupSampleOffsetsWeights(false);
		SetupPixelSize(outputviewport.Width, outputviewport.Height);
		//pContext->ClearRenderTargetView(pRTVs[0], g_XMZero.f);
		pContext->PSSetShaderResources(0, 1, pNullSRVs);
		pContext->OMSetRenderTargets(1, pRTVs, NULL);
		pContext->PSSetShaderResources(0, 1, pSRVs);
		RenderPass(pContext, 2);

		// Combination to final target
		m_SwapBuffer.SwapBuffer();
		pRTVs[0] = target.RenderTargetView();
		pContext->RSSetViewports(1, &outputviewport);
		SetupPixelSize(outputviewport.Width, outputviewport.Height);
		if (m_Mode == BloomCombination)
		{
			pSRVs[0] = source.ShaderResourceView();
			pSRVs[1] = m_SwapBuffer.ShaderResourceView();
			pRTVs[0] = target.RenderTargetView();
			pContext->OMSetRenderTargets(1, pRTVs, NULL);
			pContext->PSSetShaderResources(0, 2, pSRVs);
			RenderPass(pContext, 3);
			pContext->OMSetDepthStencilState(states.DepthDefault(), -1);
		}
		else if (m_Mode == AlphaAsDepth)
		{
			pContext->OMSetBlendState(states.AlphaBlend(), g_XMOne.f, -1);
			pContext->OMSetDepthStencilState(states.DepthDefault(), -1);
			pSRVs[0] = m_SwapBuffer.ShaderResourceView();
			pContext->OMSetRenderTargets(1, pRTVs, m_DepthStencil);
			pContext->PSSetShaderResources(0, 1, pSRVs);
			RenderPass(pContext, 4);
		} else if (m_Mode == AlphaBlendNoDepth)
		{
			pContext->OMSetBlendState(states.AlphaBlend(), g_XMOne.f, -1);
			pContext->OMSetDepthStencilState(states.DepthNone(), -1);
			pSRVs[0] = m_SwapBuffer.ShaderResourceView();
			pContext->OMSetRenderTargets(1, pRTVs, NULL);
			pContext->PSSetShaderResources(0, 1, pSRVs);
			RenderPass(pContext, 5);
		}

	}

};

class GuassianBlurEffect::Impl : public BlurEffect
{
public:
	Impl(ID3D11Device* pDevice)
		: BlurEffect(pDevice)
	{
		m_KernalRadius = 1.0f;
	}

	float GaussianDistribution(float x, float rho)
	{
		float g = 1.0f / sqrtf(XM_2PI * rho * rho);
		g *= expf(-(x * x) / (2 * rho * rho));

		return g;
	}

	//// The separable blur function uses a gaussian weighting function applied to 15 (blurKernelSpan) texels centered on the texture
	//// coordinate currently being processed. These are applied along a row (horizontal) or column (vertical). Element blurKernelMidPoint aligns with the current texel.
	//// Because the offsets and weights are symmetrical about the center texel, the offsets can be computed for one "side" and mirrored with
	//// a sign change for the other "side". Similarly the weights can be computed for one "side" and copied symmetrically to the other "side".

	void GetSampleOffsetsWeightsForBlur(
		_In_ uint textureWidthOrHeight,
		_In_ float deviation,
		_In_ float multiplier,
		uint* indicesCount,
		float* textureCoordinateOffsets,
		float4* colorWeights)
	{
		float texelS = 1.0f / static_cast<float>(textureWidthOrHeight);

		auto blurKernelMidPoint = std::max(0,std::min((int)ceilf(3 * deviation), (int)(MaxSampleCount - 1) / 2));
		auto blurKernelSpan = blurKernelMidPoint * 2 + 1;
		*indicesCount = blurKernelSpan;
		// Fill the center texel

		float weight = 1.0f * GaussianDistribution(0,deviation);
		colorWeights[blurKernelMidPoint] = float4(weight, weight, weight, weight);

		float weightsSum = weight;

		textureCoordinateOffsets[blurKernelMidPoint] = 0.0f;

		// Fill one side
		for (int i = 1; i <= blurKernelMidPoint; i++)
		{
			weight = multiplier * GaussianDistribution(static_cast<float>(i),deviation);
			textureCoordinateOffsets[blurKernelMidPoint - i] = -i * texelS;

			weightsSum += weight * 2;
			colorWeights[blurKernelMidPoint - i] = float4(weight, weight, weight, weight);
		}

		// Normalize the sum of weights to 1.0
		for (size_t i = 0; i < blurKernelMidPoint; i++)
		{
			colorWeights[blurKernelMidPoint] /= weightsSum;
		}

		// Copy to the other side
		for (int i = (blurKernelMidPoint + 1); i < blurKernelSpan; i++)
		{
			colorWeights[i] = colorWeights[(blurKernelSpan - 1) - i];
			textureCoordinateOffsets[i] = -textureCoordinateOffsets[(blurKernelSpan - 1) - i];
		}
	}

	virtual void SetupSampleOffsetsWeights(bool horizental) override;

	float m_KernalRadius;
};

DirectX::GuassianBlurEffect::GuassianBlurEffect(ID3D11Device * pDevice)
	: pImpl(new Impl(pDevice))
{
}

DirectX::GuassianBlurEffect::~GuassianBlurEffect()
{
}

void DirectX::GuassianBlurEffect::SetOutputMode(PostEffectOutputMode outputMode)
{
	pImpl->m_Mode = outputMode;
}

void DirectX::GuassianBlurEffect::SetOutputDepthStencil(ID3D11DepthStencilView * pDepthStencilBuffer)
{
	pImpl->m_DepthStencil = pDepthStencilBuffer;
}

void DirectX::GuassianBlurEffect::SetMultiplier(float multiplier)
{
	pImpl->constants.muiltiplier = multiplier;
	pImpl->dirtyFlags |= EffectDirtyFlags::PSConstantBuffer;
}

void GuassianBlurEffect::SetBlurRadius(float radius)
{
	pImpl->m_KernalRadius = radius;
}

void GuassianBlurEffect::ResizeBufferRespectTo(ID3D11Device * pDevice, const Texture2D & texture)
{
	pImpl->ResizeBuffer(pDevice, XMUINT2(texture.Width()/2, texture.Height()/2), texture.Format());
}

void GuassianBlurEffect::ResizeBuffer(ID3D11Device * pDevice, const XMUINT2 & size, DXGI_FORMAT format)
{
	pImpl->ResizeBuffer(pDevice, size, format);
}

void GuassianBlurEffect::SetAddtionalInputResources(UINT numSource, ID3D11ShaderResourceView * const * pSources)
{
}

void GuassianBlurEffect::Apply(ID3D11DeviceContext * pContext, RenderableTexture2D & target, const Texture2D & source, const D3D11_VIEWPORT & outputviewport, const D3D11_VIEWPORT & inputViewport)
{
	pImpl->Apply(pContext, target, source, outputviewport, inputViewport);
}

void GuassianBlurEffect::Impl::SetupSampleOffsetsWeights(bool horizental)
{
	size_t texture_scale = m_SwapBuffer.Width();
	if (!horizental)
		texture_scale = m_SwapBuffer.Height();

	float derivation = m_KernalRadius;
	float offsets[MaxSampleCount];

	for (size_t i = 0; i < MaxSampleCount; i++)
	{
		constants.sampleOffsets[i] = float4(0, 0, 0, 0);
		constants.sampleWeights[i] = float4(0, 0, 0, 0);
	}

	GetSampleOffsetsWeightsForBlur(texture_scale, derivation, 1.0f,
		&constants.sampleCount,
		offsets,
		constants.sampleWeights);

	if (horizental)
		for (size_t i = 0; i < constants.sampleCount; i++)
		constants.sampleOffsets[i].x = offsets[i];
	else
		for (size_t i = 0; i < constants.sampleCount; i++)
		constants.sampleOffsets[i].y = offsets[i];

	dirtyFlags |= EffectDirtyFlags::PSConstantBuffer;
}

void ChainingPostEffect::Apply(ID3D11DeviceContext * pContext, RenderableTexture2D & target, const Texture2D & source, const D3D11_VIEWPORT & outputviewport, const D3D11_VIEWPORT & inputViewport)
{
	if (target != nullptr)
	{
		//pPostEffect->Apply(pContext, target, source, outputviewport, inputViewport);
	}
}

void ChainingPostEffect::SetAddtionalInputResources(UINT numSource, ID3D11ShaderResourceView * const * pSources)
{
}