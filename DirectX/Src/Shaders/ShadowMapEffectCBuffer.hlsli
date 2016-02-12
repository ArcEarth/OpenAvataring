#define MAX_LIGHTS 4
#define MAX_BONES 72

cbuffer ShadowMapEffectCBuffer REGISTER(b0)
{
	float4x4 World;
	float4x4 ViewProjection;
	float4x4 UVTransform;
	float3   EyePosition;
	float	 Bias;

    float4   FogColor;
    float4   FogVector;

	float4	 MaterialAmbient;
	float4	 MaterialDiffuse;
	float3   MaterialSpecular;
	float    MaterialSpecularPower;
	float4   MaterialEmissive;

	float4	 AmbientLight;
	float4   LightColor[MAX_LIGHTS];
	float4	 LightAttenuation[MAX_LIGHTS];
	float4	 LightPosition[MAX_LIGHTS];
	float4x4 LightViewProjection[MAX_LIGHTS];
	float4	 LightDirection[MAX_LIGHTS];
    float    LightBias[MAX_LIGHTS];
	uint     IsPointLight[MAX_LIGHTS];
};
