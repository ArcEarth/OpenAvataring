// A constant buffer that stores the three basic column-major matrices for composing geometry.
cbuffer ModelViewProjectionConstantBuffer : register(b0)
{
	matrix WorldViewProj;
	float4 ShadowColor;
};

Texture2D gDiffuseMap : register(t0);
SamplerState samLinear : register(s0);

struct VSInput
{
	float4 Position : SV_Position;
};

struct PSInput
{
	float4 Position : SV_Position;
	float3 EnvCoord : TEXCOORD0;
}

PSInput VS(VSInputWeights vin) : SV_POSITION
{
	PSInput vout;
	vout.EnvCoord = vin.Position;
	vout.Position = mul(vin.Position,WorldViewProj)
	return vout;
}

float4 PS(PSInput pixel) : SV_TARGET
{
	float4 diffuse = gDiffuseMap.Sample(samLinear, pixel.EnvCoord);
	return diffuse;
}
