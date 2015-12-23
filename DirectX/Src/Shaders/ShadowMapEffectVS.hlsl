#define REGISTER(b) : register(b)

#include "ShadowMapEffectCBuffer.hlsli"

cbuffer SkinningBuffer : register (b1)
{
	float4x3 Bones[MAX_BONES];
};

#include "ShadowMapEffectStructures.hlsli"

struct VSInputNoTex
{
	float4 Position : SV_Position;
	float3 Normal	: NORMAL;
};

struct VSInputTex
{
	float4 Position : SV_Position;
	float3 Normal	: NORMAL;
	float2 TexCoord : TEXCOORD0;
};

struct VSInputNoTexWeights
{
	float4 Position : SV_Position;
	float3 Normal	: NORMAL;
	uint4  Indices  : BLENDINDICES0;
	float4 Weights  : BLENDWEIGHT0;
};

struct VSInputTexWeights
{
	float4 Position : SV_Position;
	float3 Normal	: NORMAL;
	float2 TexCoord : TEXCOORD0;
	uint4  Indices  : BLENDINDICES0;
	float4 Weights  : BLENDWEIGHT0;
};

struct VSInputTexTangentWeights
{
	float4 Position : SV_Position;
	float3 Normal	: NORMAL;
	float3 Tangent	: TANGENT;
	float2 TexCoord : TEXCOORD0;
	uint4  Indices  : BLENDINDICES0;
	float4 Weights  : BLENDWEIGHT0;
};

#include "Common.hlsli"

void SkinVertexNoTex(inout VSInputNoTexWeights vin, uniform int boneCount)
{
	SkinVertexWithNormal;
}

void SkinVertexTex(inout VSInputTexWeights vin, uniform int boneCount)
{
	SkinVertexWithNormal;
}

void SkinVertexTexBump(inout VSInputTexTangentWeights vin, uniform int boneCount)
{
	SkinVertexWithNormalTagent;
}


PSInputOneLightNoTex VS_OneLightNoBoneNoTex(VSInputNoTex vin)
{
	PSInputOneLightNoTex vout;

	SetPositionNormalToEye;

	SetLightUVOne;

	return vout;
}

PSInputOneLightTex VS_OneLightNoBoneTex(VSInputTex vin)
{
	PSInputOneLightTex vout;

	SetPositionNormalToEye;

	SetTextureCoord;

	SetLightUVOne;

	return vout;
}

PSInputOneLightTexBump VS_OneLightNoBoneTexBump(VSInputTexTangentWeights vin)
{
    PSInputOneLightTexBump vout;

	SetPositionNormalToEyeTangentBinormal;

	SetTextureCoord;

	SetLightUVOne;

    return vout;
}

PSInputOneLightNoTex VS_OneLightOneBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputOneLightNoTex vout;

	SkinVertexNoTex(vin, 1);

	SetPositionNormalToEye;

	SetLightUVOne;

	return vout;
}

PSInputOneLightTex VS_OneLightOneBoneTex(VSInputTexWeights vin)
{
	PSInputOneLightTex vout;

	SkinVertexTex(vin, 1);

	SetPositionNormalToEye;

	SetTextureCoord;

	SetLightUVOne;

	return vout;
}

PSInputOneLightNoTex VS_OneLightTwoBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputOneLightNoTex vout;

	SkinVertexNoTex(vin, 2);

	SetPositionNormalToEye;

	SetLightUVOne;

	return vout;
}

PSInputOneLightTex VS_OneLightTwoBoneTex(VSInputTexWeights vin)
{
	PSInputOneLightTex vout;

	SkinVertexTex(vin, 2);

	SetPositionNormalToEye;

	SetTextureCoord;

	SetLightUVOne;

	return vout;
}

PSInputOneLightNoTex VS_OneLightFourBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputOneLightNoTex vout;

	SkinVertexNoTex(vin, 4);

	SetPositionNormalToEye;

	SetLightUVOne;

	return vout;
}

PSInputOneLightTex VS_OneLightFourBoneTex(VSInputTexWeights vin)
{
	PSInputOneLightTex vout;

	SkinVertexTex(vin, 4);

	SetPositionNormalToEye;

	SetTextureCoord;

	SetLightUVOne;

	return vout;
}

PSInputOneLightTexBump VS_OneLightFourBoneTexBump(VSInputTexTangentWeights vin)
{
    PSInputOneLightTexBump vout;

    SkinVertexTexBump(vin, 4);

	SetPositionNormalToEyeTangentBinormal;

	SetTextureCoord;

	SetLightUVOne;

    return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceNoBoneTex(VSInputTex vin)
{
	PSInputScreenSpaceTex vout;

	SetPositionNormalToEye;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceOneBoneTex(VSInputTexWeights vin)
{
	PSInputScreenSpaceTex vout;

	SkinVertexTex(vin, 1);

	SetPositionNormalToEye;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceTwoBoneTex(VSInputTexWeights vin)
{
	PSInputScreenSpaceTex vout;

	SkinVertexTex(vin, 2);

	SetPositionNormalToEye;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceFourBoneTex(VSInputTexWeights vin)
{
	PSInputScreenSpaceTex vout;

	SkinVertexTex(vin, 4);

	SetPositionNormalToEye;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}


PSInputScreenSpaceTex VS_ScreenSpaceNoBoneTexBump(VSInputTexTangentWeights vin)
{
	PSInputScreenSpaceTex vout;

	SetPositionNormalToEyeTangentBinormal;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceOneBoneTexBump(VSInputTexTangentWeights vin)
{
	PSInputScreenSpaceTex vout;

	SkinVertexTexBump(vin, 1);

	SetPositionNormalToEyeTangentBinormal;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceTwoBoneTexBump(VSInputTexTangentWeights vin)
{
	PSInputScreenSpaceTex vout;

	SkinVertexTexBump(vin, 2);

	SetPositionNormalToEyeTangentBinormal;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceTex VS_ScreenSpaceFourBoneTexBump(VSInputTexTangentWeights vin)
{
	PSInputScreenSpaceTex vout;

	SkinVertexTexBump(vin, 4);

	SetPositionNormalToEyeTangentBinormal;

	SetPositionUV;

	SetTextureCoord;

	return vout;
}

PSInputScreenSpaceNoTex VS_ScreenSpaceNoBoneNoTex(VSInputNoTex vin)
{
	PSInputScreenSpaceNoTex vout;

	SetPositionNormalToEye;

	SetPositionUV;

	return vout;
}

PSInputScreenSpaceNoTex VS_ScreenSpaceOneBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputScreenSpaceNoTex vout;

	SkinVertexNoTex(vin, 1);

	SetPositionNormalToEye;

	SetPositionUV;

	return vout;
}

PSInputScreenSpaceNoTex VS_ScreenSpaceTwoBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputScreenSpaceNoTex vout;

	SkinVertexNoTex(vin, 2);

	SetPositionNormalToEye;

	SetPositionUV;

	return vout;
}

PSInputScreenSpaceNoTex VS_ScreenSpaceFourBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputScreenSpaceNoTex vout;

	SkinVertexNoTex(vin, 4);

	SetPositionNormalToEye;

	SetPositionUV;

	return vout;
}

PSInputBinaryOneLightNoTex VS_BinaryOneLightNoBoneNoTex(VSInputNoTex vin)
{
	PSInputBinaryOneLightNoTex vout;

	SetPosition;

	vout.lightUv[0] = GetLightUV(posWorld, LightViewProjection[0]);

	return vout;
}

PSInputBinaryOneLightTex VS_BinaryOneLightNoBoneTex(VSInputTex vin)
{
	PSInputBinaryOneLightTex vout;

	SetPosition;

	SetTextureCoord;

	vout.lightUv[0] = GetLightUV(posWorld, LightViewProjection[0]);

	return vout;
}

PSInputBinaryOneLightNoTex VS_BinaryOneLightFourBoneNoTex(VSInputNoTexWeights vin)
{
	PSInputBinaryOneLightNoTex vout;

	SkinVertexNoTex(vin, 4);

	SetPosition;

	vout.lightUv[0] = GetLightUV(posWorld, LightViewProjection[0]);

	return vout;
}

PSInputBinaryOneLightTex VS_BinaryOneLightFourBoneTex(VSInputTexWeights vin)
{
	PSInputBinaryOneLightTex vout;

	SkinVertexTex(vin, 4);

	SetPosition;

	SetTextureCoord;

	vout.lightUv[0] = GetLightUV(posWorld, LightViewProjection[0]);

	return vout;
}

