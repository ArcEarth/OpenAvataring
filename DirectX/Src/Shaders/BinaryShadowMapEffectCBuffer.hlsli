#define MAX_LIGHTS 4
#define MAX_BONES 72

cbuffer BinaryShadowMapEffectCBuffer REGISTER(b0)
{
	float4x4 World;
	float4x4 ViewProjection;
	float4x4 LightViewProjection[MAX_LIGHTS];
	float4x3 Bones[MAX_BONES];
};
