// A constant buffer that stores the three basic column-major matrices for composing geometry.
cbuffer ModelViewProjectionConstantBuffer : register(b0)
{
	matrix WorldViewProj;
    float4 DiffuseColor;
    float4 EmissiveColor;
};

TextureCube envMap : register(t0);
SamplerState samLinear : register(s0);

struct VSInput
{
	float4 Position : SV_Position;
};

struct PSInput
{
    float4 Position : SV_Position;
    float3 EnvCoord : TEXCOORD0;
};

// Vertex Shader
PSInput VS(VSInput vin)
{
	PSInput vout;
	vout.EnvCoord = vin.Position.xyz;
    vout.Position = mul(vin.Position, WorldViewProj);
	return vout;
}

// Pixel Shader
float4 PS(PSInput pixel) : SV_TARGET
{
    float4 diffuse = envMap.Sample(samLinear, pixel.EnvCoord);
	return diffuse;
}
