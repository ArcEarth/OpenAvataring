#include "pch_directX.h"

#include <ShadowMapGenerationEffect.h>
#include "EffectCommon.h"
#include <CommonStates.h>

using namespace DirectX;

using namespace DirectX::HLSLVectors;
XM_ALIGNATTR
struct ShadowMapGenerationEffectConstants
{
	float4x4	WorldLightProj;
	float4		ShadowColor;
	float4x3	Bones[ShadowMapGenerationEffect::MaxBones];
};

namespace DirectX
{
	namespace EffectDirtyFlags
	{
		const int BoneTransforms = 0x100;
		const int BoneColors = 0x200;
	}
}


XM_ALIGNATTR
struct BoneColorsConstants
{
	Color		BoneColors[ShadowMapGenerationEffect::MaxBones];
};

struct ShadowMapGenerationEffectTraits
{
	typedef ShadowMapGenerationEffectConstants ConstantBufferType;

	static const int VertexShaderCount = 11;
	static const int PixelShaderCount = 5;
	static const int ShaderPermutationCount = 16 + 3;
};

typedef ShadowMapGenerationEffectTraits			EffectTraitsType;
typedef EffectBase<EffectTraitsType>			EffectBaseType;

SharedResourcePool<ID3D11Device*, EffectBaseType::DeviceResources> EffectBaseType::deviceResourcesPool;

using Microsoft::WRL::ComPtr;

_MM_ALIGN16
class ShadowMapGenerationEffect::Impl : public EffectBaseType
{
public:
	BoneColorsConstants boneColors;
	int weightsPerVertex;
	ShadowFillMode fillMode;
	Matrix4x4 World, LightView, LightProj;
	ID3D11DepthStencilView* pDepthMap;
	ID3D11RenderTargetView* pRenderTargetView;
	CommonStates			commonStates;
	ConstantBuffer<BoneColorsConstants> boneColorsBuffer;

	typedef EffectTraitsType	Traits;
	typedef EffectBaseType		Base;

	Impl(ID3D11Device* device)
			: EffectBase(device),
			commonStates(device),
			weightsPerVertex(0),
			fillMode(DepthFill),
			pRenderTargetView(NULL),
			pDepthMap(NULL),
			boneColorsBuffer(device)
		{
			auto addr = (uintptr_t)&constants.WorldLightProj;
			assert((addr & 0xf) == 0);

			static_assert(_countof(Base::VertexShaderIndices) == Traits::ShaderPermutationCount, "array/max mismatch");
			static_assert(_countof(Base::VertexShaderBytecode) == Traits::VertexShaderCount, "array/max mismatch");
			static_assert(_countof(Base::PixelShaderBytecode) == Traits::PixelShaderCount, "array/max mismatch");
			static_assert(_countof(Base::PixelShaderIndices) == Traits::ShaderPermutationCount, "array/max mismatch");

			XMMATRIX id = XMMatrixIdentity();

			for (size_t i = 0; i < MaxBones; ++i)
			{
				XMStoreFloat3x4(&constants.Bones[i], id);
			}
		}

	int GetCurrentShaderPermutation() const
	{
		int perm = 0;
		// 0 1 2 4
		if (weightsPerVertex <= 2) perm = weightsPerVertex;
		if (weightsPerVertex == 4) perm = 3;

		if (texture != nullptr)
			perm += 4;

		if (fillMode == SolidColorFill || (weightsPerVertex == 0 && fillMode == BoneColorFill))
			perm += 8;
		else if (weightsPerVertex > 0 && fillMode == BoneColorFill)
		{
			//assert(weightsPerVertex != 0);
			perm = 16;
			if (weightsPerVertex <= 2) perm += weightsPerVertex - 1;
			if (weightsPerVertex == 4) perm += 2;
		}

		return perm;
	}

	void Apply(ID3D11DeviceContext* deviceContext)
	{
		// Compute derived parameter values.
		matrices.SetConstants(dirtyFlags, reinterpret_cast<XMMATRIX&>(constants.WorldLightProj));

		if (dirtyFlags & EffectDirtyFlags::AlphaTest && texture != nullptr)
		{
			ID3D11ShaderResourceView* pSrvs[] = { texture.Get()/*, NULL, NULL, NULL, NULL, NULL*/ };
			deviceContext->PSSetShaderResources(0, 1, pSrvs);
			auto pSampler = commonStates.LinearWrap();
			deviceContext->PSSetSamplers(0, 1, &pSampler);
			dirtyFlags &= ~EffectDirtyFlags::AlphaTest;
		}

		if (fillMode == BoneColorFill && dirtyFlags & EffectDirtyFlags::BoneColors)
		{
			boneColorsBuffer.SetData(deviceContext, boneColors);
			auto pBuffer = boneColorsBuffer.GetBuffer();
			deviceContext->VSSetConstantBuffers(1, 1, &pBuffer);
		}

		// Set the render targets if applicatable
		if (pDepthMap != nullptr || pRenderTargetView != nullptr)
		{
			//ID3D11ShaderResourceView* pSrvs[] = {NULL, NULL, NULL, NULL, NULL, NULL};
			//deviceContext->PSSetShaderResources(0, sizeof(pSrvs), pSrvs);
			assert(fillMode != SolidColorFill && pRenderTargetView != nullptr);
			ID3D11RenderTargetView* rtvs[] = { pRenderTargetView };
			deviceContext->OMSetRenderTargets(1, rtvs, pDepthMap);
		}

		// Set Blend and depth state
		deviceContext->OMSetDepthStencilState(commonStates.DepthDefault(), -1);
		deviceContext->OMSetBlendState(commonStates.Opaque(), g_XMOne.f, -1);

		ApplyShaders(deviceContext, GetCurrentShaderPermutation());

	}
};


namespace
{
#if defined(_XBOX_ONE) && defined(_TITLE)
#include "Shaders/Xbox/ShadowMapGen_VS_NoBone.inc"
#include "Shaders/Xbox/ShadowMapGen_VS_OneBone.inc"
#include "Shaders/Xbox/ShadowMapGen_VS_TwoBone.inc"
#include "Shaders/Xbox/ShadowMapGen_VS_FourBone.inc"

#include "Shaders/Windows/ShadowMapGen_PS.inc"
#else
#include "Shaders/Windows/ShadowMapGen_VS_NoBone.inc"
#include "Shaders/Windows/ShadowMapGen_VS_OneBone.inc"
#include "Shaders/Windows/ShadowMapGen_VS_TwoBone.inc"
#include "Shaders/Windows/ShadowMapGen_VS_FourBone.inc"
#include "Shaders/Windows/ShadowMapGen_VS_NoBoneTex.inc"
#include "Shaders/Windows/ShadowMapGen_VS_OneBoneTex.inc"
#include "Shaders/Windows/ShadowMapGen_VS_TwoBoneTex.inc"
#include "Shaders/Windows/ShadowMapGen_VS_FourBoneTex.inc"

#include "Shaders/windows/ShadowMapGen_VS_OneBoneColor.inc"
#include "Shaders/windows/ShadowMapGen_VS_TwoBoneColor.inc"
#include "Shaders/windows/ShadowMapGen_VS_FourBoneColor.inc"

#include "Shaders/Windows/ShadowMapGen_PS_DepthNoTex.inc"
#include "Shaders/Windows/ShadowMapGen_PS_DepthTex.inc"
#include "Shaders/Windows/ShadowMapGen_PS_ColorNoTex.inc"
#include "Shaders/Windows/ShadowMapGen_PS_ColorTex.inc"

#include "Shaders/windows/ShadowMapGen_PS_VertexColor.inc"
#endif
}

template <size_t Size>
inline ShaderBytecode MakeShaderByteCode(const BYTE(&bytecode)[Size])
{
	return ShaderBytecode{ bytecode ,sizeof(bytecode) };
}

const ShaderBytecode EffectBase<ShadowMapGenerationEffectTraits>::VertexShaderBytecode[] =
{
	MakeShaderByteCode(ShadowMapGen_VS_NoBone),
	MakeShaderByteCode(ShadowMapGen_VS_OneBone),
	MakeShaderByteCode(ShadowMapGen_VS_TwoBone),
	MakeShaderByteCode(ShadowMapGen_VS_FourBone),
	MakeShaderByteCode(ShadowMapGen_VS_NoBoneTex),
	MakeShaderByteCode(ShadowMapGen_VS_OneBoneTex),
	MakeShaderByteCode(ShadowMapGen_VS_TwoBoneTex),
	MakeShaderByteCode(ShadowMapGen_VS_FourBoneTex),
	MakeShaderByteCode(ShadowMapGen_VS_OneBoneColor),
	MakeShaderByteCode(ShadowMapGen_VS_TwoBoneColor),
	MakeShaderByteCode(ShadowMapGen_VS_FourBoneColor),
};


const int EffectBase<ShadowMapGenerationEffectTraits>::VertexShaderIndices[] =
{
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,

	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,

	8,
	9,
	10,
};


const ShaderBytecode EffectBase<ShadowMapGenerationEffectTraits>::PixelShaderBytecode[] =
{
	MakeShaderByteCode(ShadowMapGen_PS_DepthNoTex),
	MakeShaderByteCode(ShadowMapGen_PS_DepthTex),
	MakeShaderByteCode(ShadowMapGen_PS_ColorNoTex),
	MakeShaderByteCode(ShadowMapGen_PS_ColorTex),
	MakeShaderByteCode(ShadowMapGen_PS_VertexColor),
};


const int EffectBase<ShadowMapGenerationEffectTraits>::PixelShaderIndices[] =
{
	0,
	0,
	0,
	0,
	1,
	1,
	1,
	1,

	2,
	2,
	2,
	2,
	3,
	3,
	3,
	3,

	4,
	4,
	4,
};

ShadowMapGenerationEffect::ShadowMapGenerationEffect(ID3D11Device * device)
	: pImpl(new Impl(device))
{
}

ShadowMapGenerationEffect::~ShadowMapGenerationEffect()
{
}


void ShadowMapGenerationEffect::SetShadowMap(ID3D11DepthStencilView * pShaodwMap, ID3D11RenderTargetView* pRTV)
{
	pImpl->pRenderTargetView = pRTV;
	pImpl->pDepthMap = pShaodwMap;
}

ShadowMapGenerationEffect::ShadowFillMode ShadowMapGenerationEffect::GetShadowFillMode() const
{
	return pImpl->fillMode;
}

void XM_CALLCONV ShadowMapGenerationEffect::SetShadowFillMode(ShadowFillMode mode)
{
	pImpl->fillMode = mode;
}

void XM_CALLCONV ShadowMapGenerationEffect::SetShadowColor(FXMVECTOR color)
{
	pImpl->constants.ShadowColor = color;
	pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}

void ShadowMapGenerationEffect::SetBoneColors(XMVECTOR const * value, size_t count)
{
	std::copy_n(reinterpret_cast<const Color*>(value), std::min(count, MaxBones), pImpl->boneColors.BoneColors);
	pImpl->dirtyFlags |= EffectDirtyFlags::BoneColors;
}

void ShadowMapGenerationEffect::SetAlphaDiscardThreshold(float clipThreshold)
{
}

void ShadowMapGenerationEffect::SetAlphaDiscardTexture(ID3D11ShaderResourceView * pTexture)
{
	if (pImpl->texture.Get() == pTexture)
		return;
	pImpl->texture = pTexture;
	pImpl->dirtyFlags |= EffectDirtyFlags::AlphaTest;
}

void ShadowMapGenerationEffect::DisableAlphaDiscard()
{
	if (pImpl->texture == nullptr)
		return;
	pImpl->texture = nullptr;
	pImpl->dirtyFlags |= EffectDirtyFlags::AlphaTest;
}

void XM_CALLCONV ShadowMapGenerationEffect::SetWorld(FXMMATRIX value)
{
	pImpl->matrices.world = value;

	pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::WorldInverseTranspose | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV ShadowMapGenerationEffect::SetView(FXMMATRIX value)
{
	pImpl->matrices.view = value;

	pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj | EffectDirtyFlags::EyePosition | EffectDirtyFlags::FogVector;
}


void XM_CALLCONV ShadowMapGenerationEffect::SetProjection(FXMMATRIX value)
{
	pImpl->matrices.projection = value;

	pImpl->dirtyFlags |= EffectDirtyFlags::WorldViewProj;
}

void ShadowMapGenerationEffect::SetWeightsPerVertex(int value)
{
	if ((value != 0) &&
		(value != 1) &&
		(value != 2) &&
		(value != 4))
	{
		throw std::out_of_range("WeightsPerVertex must be 0, 1, 2, or 4");
	}

	pImpl->weightsPerVertex = value;
}


void ShadowMapGenerationEffect::SetBoneTransforms(_In_reads_(count) XMMATRIX const* value, size_t count)
{
	if (count > MaxBones)
		throw std::out_of_range("count parameter out of range");

	auto& boneConstant = pImpl->constants.Bones;

	for (size_t i = 0; i < count; i++)
	{
		XMMATRIX boneMatrix = XMMatrixTranspose(value[i]);
		XMStoreFloat3x4(&boneConstant[i], boneMatrix);
	}

	pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}


void ShadowMapGenerationEffect::ResetBoneTransforms()
{
	auto boneConstant = pImpl->constants.Bones;

	XMMATRIX id = XMMatrixIdentity();

	for (size_t i = 0; i < MaxBones; ++i)
	{
		XMStoreFloat3x4(&boneConstant[i], id);
	}

	pImpl->dirtyFlags |= EffectDirtyFlags::ConstantBuffer;
}



void ShadowMapGenerationEffect::Apply(ID3D11DeviceContext * deviceContext)
{
	pImpl->Apply(deviceContext);
	//ApplyShaders(deviceContext, m_pImpl->GetCurrentShaderPermutation());
}

void ShadowMapGenerationEffect::GetVertexShaderBytecode(void const ** pShaderByteCode, size_t * pByteCodeLength)
{
	auto perm = pImpl->GetCurrentShaderPermutation();
	pImpl->GetVertexShaderBytecode(perm, pShaderByteCode, pByteCodeLength);
}
