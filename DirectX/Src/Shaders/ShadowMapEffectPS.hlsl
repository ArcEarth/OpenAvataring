Texture2D DiffuseTex : register(t0);
Texture2D<float2> NormalTex : register(t1);
Texture2D SpecularTex : register(t2);
Texture2D ShadowTex : register(t3);

// linear 
SamplerState MaterialSampler : register(s0);
SamplerComparisonState ShadowSampleCmpState : register(s1);
SamplerState ShadowSampler : register(s2);

#define REGISTER(b) : register(b)

#include "ShadowMapEffectCBuffer.hlsli"

#include "ShadowMapEffectStructures.hlsli"

//
// lambert lighting function
//
float3 LambertLighting(
	float3 lightNormal,
	float3 surfaceNormal,
	float3 lightColor,
	float3 pixelColor
	)
{
	// compute amount of contribution per light
	float diffuseAmount = saturate(dot(lightNormal, surfaceNormal));
	float3 diffuse = diffuseAmount * lightColor * pixelColor;
	return diffuse;
}

void NormalizeLightUV(inout float4 luv)
{
    luv = luv / luv.w; // Normalize homogenous coords
    luv.xy = float2(0.5f, -0.5f) * luv.xy + 0.5f; // Projection space to UV space transform
}

//
// specular contribution function
//
float3 SpecularContribution(
	float3 toEye,
	float3 lightNormal,
	float3 surfaceNormal,
	float3 materialSpecularColor,
	float  materialSpecularPower,
	float  lightSpecularIntensity,
	float3 lightColor)
{
	// compute specular contribution
	float3 vHalf = normalize(lightNormal + toEye);
	float specularAmount = saturate(dot(surfaceNormal, vHalf));
	specularAmount = pow(specularAmount, materialSpecularPower) * lightSpecularIntensity;
	float3 specular = materialSpecularColor * lightColor * specularAmount;

	return specular;
}

float SampleShadow(Texture2D tex, float2 uv, float depth)
{
	float shadowAmount = 0;
	shadowAmount =  tex.SampleCmpLevelZero(ShadowSampleCmpState, uv, depth, int2(-1, -1)).r;
	shadowAmount += tex.SampleCmpLevelZero(ShadowSampleCmpState, uv, depth, int2(1, -1)).r;
	shadowAmount += tex.SampleCmpLevelZero(ShadowSampleCmpState, uv, depth, int2(-1, 1)).r;
	shadowAmount += tex.SampleCmpLevelZero(ShadowSampleCmpState, uv, depth, int2(1, 1)).r;
	shadowAmount /= 4.0f;
	return shadowAmount;
}

float4 PS_OneLightNoTex(PSInputOneLightNoTex pixel) : SV_TARGET
{
	float3 worldNormal = normalize(pixel.normal);
	float3 toEyeVector = normalize(pixel.toEye);

	float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb;
	float3 matDiffuse = MaterialDiffuse.rgb;
	float3 specular = 0;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
        NormalizeLightUV(pixel.lightUv[i]);
        float shadowAmount = SampleShadow(ShadowTex, pixel.lightUv[i].xy, pixel.lightUv[i].z - Bias);
		diffuse += shadowAmount * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
	}
	diffuse = saturate(diffuse + specular);

	return float4(diffuse, MaterialDiffuse.a);
}

float4 PS_OneLightTex(PSInputOneLightTex pixel) : SV_TARGET
{
    float4 texDiffuse = DiffuseTex.Sample(MaterialSampler, pixel.uv);

	clip(texDiffuse.a - 0.15);

	float3 worldNormal = normalize(pixel.normal);
	float3 toEyeVector = normalize(pixel.toEye);

	float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb * texDiffuse.rgb;
	float3 matDiffuse = MaterialDiffuse.rgb * texDiffuse.rgb;

	float3 specular = 0;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
        NormalizeLightUV(pixel.lightUv[i]);
        float shadowAmount = SampleShadow(ShadowTex, pixel.lightUv[i].xy, pixel.lightUv[i].z - Bias);
		//float shadowAmount = 1.0f;
		diffuse += shadowAmount * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
		specular += shadowAmount * SpecularContribution(toEyeVector, LightDirection[i].xyz, worldNormal, MaterialSpecular.rgb, MaterialSpecularPower, LightColor[i].a, LightColor[i].rgb);
	}

	diffuse = saturate(diffuse + specular);

	return float4(diffuse, MaterialDiffuse.a * texDiffuse.a);
}

float4 PS_OneLightTexBump(PSInputOneLightTexBump pixel) : SV_TARGET
{
	float4 texDiffuse = DiffuseTex.Sample(MaterialSampler, pixel.uv).rgba;

	clip(texDiffuse.a - 0.15);

	float3 toEyeVector = normalize(pixel.toEye);

	// Sample the pixel in the bump map.
	float2 texBump = NormalTex.Sample(MaterialSampler, pixel.uv);
	texBump = (texBump * 2.0f) - 1.0f;
	// Calculate the normal from the data in the bump map.
	float3 worldNormal = pixel.normal + texBump.x * pixel.tangent + texBump.y * pixel.binormal;
	worldNormal = normalize(worldNormal);


	float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb * texDiffuse.rgb;
	float3 matDiffuse = MaterialDiffuse.rgb * texDiffuse.rgb;

	float3 specular = 0;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
        NormalizeLightUV(pixel.lightUv[i]);
        float shadowAmount = SampleShadow(ShadowTex, pixel.lightUv[i].xy, pixel.lightUv[i].z - Bias);
		diffuse += shadowAmount * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
		specular += shadowAmount * SpecularContribution(toEyeVector, LightDirection[i].xyz, worldNormal, MaterialSpecular.rgb, MaterialSpecularPower, LightColor[i].a, LightColor[i].rgb);
	}
	diffuse = saturate(diffuse + specular);

	return float4(diffuse, MaterialDiffuse.a * texDiffuse.a);
}

float4 PS_OneLightTexBumpSpecular(PSInputOneLightTexBump pixel) : SV_TARGET
{
    float4 texDiffuse = DiffuseTex.Sample(MaterialSampler, pixel.uv).rgba;

    clip(texDiffuse.a - 0.15);

    float3 toEyeVector = normalize(pixel.toEye);

	// Sample the pixel in the bump map.
    float2 texBump = NormalTex.Sample(MaterialSampler, pixel.uv);
    texBump = (texBump * 2.0f) - 1.0f;
	// Calculate the normal from the data in the bump map.
    float3 worldNormal = pixel.normal + texBump.x * pixel.tangent + texBump.y * pixel.binormal;
    worldNormal = normalize(worldNormal);


    float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb * texDiffuse.rgb;
    float3 matDiffuse = MaterialDiffuse.rgb * texDiffuse.rgb;

    float4 matSpecular = SpecularTex.Sample(MaterialSampler, pixel.uv);
    float3 specular = matSpecular * AmbientLight.rgb;

	[unroll]
    for (int i = 0; i < 1; i++)
    {
        NormalizeLightUV(pixel.lightUv[i]);
        float shadowAmount = SampleShadow(ShadowTex, pixel.lightUv[i].xy, pixel.lightUv[i].z - Bias);
        diffuse += shadowAmount * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
        specular += shadowAmount * SpecularContribution(toEyeVector, LightDirection[i].xyz, worldNormal, matSpecular.rgb, MaterialSpecularPower, LightColor[i].a, LightColor[i].rgb);
    }
    diffuse = saturate(diffuse + specular);

    return float4(diffuse, MaterialDiffuse.a * texDiffuse.a);
}



float4 PS_ScreenSpaceNoTex(PSInputScreenSpaceNoTex pixel) : SV_TARGET
{
	float3 worldNormal = normalize(pixel.normal);
	float3 toEyeVector = normalize(pixel.toEye);

	float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb;
	float3 matDiffuse = MaterialDiffuse.rgb;
	float3 specular = 0;

	//float2 pixeluv = pixel.pos.xy * float2(0.5f, -0.5f) + 0.5f;
	//float4 shadowAmount = ShadowTex.Sample(ShadowSampler, pixel.posUV.xy);
	float4 shadowAmount = ShadowTex.Load(pixel.pos).rgba;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
		diffuse += shadowAmount[i] * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
		specular += shadowAmount[i] * SpecularContribution(toEyeVector, LightDirection[i].xyz, worldNormal, MaterialSpecular.rgb, MaterialSpecularPower, LightColor[i].a, LightColor[i].rgb);
	}
	diffuse = saturate(diffuse + specular);

	return float4(diffuse, MaterialDiffuse.a);
}

float4 PS_ScreenSpaceTex(PSInputScreenSpaceTex pixel) : SV_TARGET
{
	float4 texDiffuse = DiffuseTex.Sample(MaterialSampler, pixel.uv);

	clip(texDiffuse.a - 0.15);

	float3 worldNormal = normalize(pixel.normal);
	float3 toEyeVector = normalize(pixel.toEye);


	float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb * texDiffuse.rgb;
	float3 matDiffuse = MaterialDiffuse.rgb * texDiffuse.rgb;

	float4 shadowAmount = ShadowTex.Load(pixel.pos);
	float3 specular = .0f;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
		diffuse += shadowAmount[i] * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
		specular += shadowAmount[i] * SpecularContribution(toEyeVector, LightDirection[i].xyz, worldNormal, MaterialSpecular.rgb, MaterialSpecularPower, LightColor[i].a, LightColor[i].rgb);
	}
	diffuse = saturate(diffuse + specular);

	return float4(diffuse, MaterialDiffuse.a * texDiffuse.a);
}

float4 PS_ScreenSpaceTexBump(PSInputScreenSpaceTex pixel) : SV_TARGET
{
	float4 texDiffuse = DiffuseTex.Sample(MaterialSampler, pixel.uv);

	clip(texDiffuse.a - 0.15);

	// Sample the pixel in the bump map.
	float2 texBump = NormalTex.Sample(MaterialSampler, pixel.uv).rg;
	texBump = (texBump * 2.0f) - 1.0f;
	// Calculate the normal from the data in the bump map.
	float3 worldNormal = pixel.normal + texBump.x * pixel.tangent + texBump.y * pixel.binormal;
	worldNormal = normalize(worldNormal);

	float3 toEyeVector = normalize(pixel.toEye);


	float3 diffuse = MaterialAmbient.rgb * AmbientLight.rgb * texDiffuse.rgb;
	float3 matDiffuse = MaterialDiffuse.rgb * texDiffuse.rgb;

	float4 shadowAmount = ShadowTex.Load(pixel.pos).rgba;
	float3 specular = .0f;																	   //shadowAmount = abs(pixel.pos.z - shadowAmount.w) < Bias ? shadowAmount : 1.0f;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
		diffuse += shadowAmount[i] * LambertLighting(LightDirection[i].xyz, worldNormal, LightColor[i].rgb, matDiffuse);
		specular += shadowAmount[i] * SpecularContribution(toEyeVector, LightDirection[i].xyz, worldNormal, MaterialSpecular.rgb, MaterialSpecularPower, LightColor[i].a, LightColor[i].rgb);
	}
	diffuse = saturate(diffuse + specular);

	return float4(diffuse, MaterialDiffuse.a * texDiffuse.a);
}


float4 PS_BinaryOneLightNoTex(PSInputBinaryOneLightNoTex pixel) : SV_TARGET
{
	float4 amount;
	amount.w = pixel.pos.z;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
        NormalizeLightUV(pixel.lightUv[i]);
		float shadowDepth = ShadowTex.Sample(ShadowSampler, pixel.lightUv[i].xy).r;
        amount[i] = (shadowDepth + Bias > pixel.lightUv[i].z) ? 1.0f : 0.0f;
    }

	//clip(0.85 - amount.r);

	return amount;
}

float4 PS_BinaryOneLightTex(PSInputBinaryOneLightTex pixel) : SV_TARGET
{
	float4 texDiffuse = DiffuseTex.Sample(MaterialSampler, pixel.uv);

	clip(texDiffuse.a - 0.15f); // Sample Alpha clip texture
	float alpha = texDiffuse.a;

	float4 amount;
	amount.w = pixel.pos.z;

	[unroll]
	for (int i = 0; i < 1; i++)
	{
        NormalizeLightUV(pixel.lightUv[i]);
		float shadowDepth = ShadowTex.Sample(ShadowSampler, pixel.lightUv[i].xy).r;
        amount[i] = (shadowDepth + Bias > pixel.lightUv[i].z) ? alpha : 0.0f;
    }

	//clip(0.85 - amount.r);

	return amount;
}
