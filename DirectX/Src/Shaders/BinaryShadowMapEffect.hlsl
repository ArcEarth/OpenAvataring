// A constant buffer that stores the three basic column-major matrices for composing geometry.
#define REGISTER(b) : register(b)

#include "BinaryShadowMapEffectCBuffer.hlsli"

Texture2D gDiffuseMap : register(t0);
SamplerState samLinear : register(s0);

struct VSInputTex
{
	float4 Position : SV_Position;
	float2 TexCoord : TEXCOORD0;
};

struct VSInputWeights
{
	float4 Position : SV_Position;
	uint4  Indices  : BLENDINDICES0;
	float4 Weights  : BLENDWEIGHT0;
};

struct VSInputTexWeights
{
	float4 Position : SV_Position;
	float2 TexCoord : TEXCOORD0;
	uint4  Indices  : BLENDINDICES0;
	float4 Weights  : BLENDWEIGHT0;
};

#include "Common.hlsli"

void SkinVertexNoTex(inout VSInputWeights vin, uniform int boneCount)
{
	SkinVertex;
}

void SkinVertexTex(inout VSInputTexWeights vin, uniform int boneCount)
{
	SkinVertex;
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
