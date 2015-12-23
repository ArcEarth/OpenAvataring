float2 PositionToUV(float4 posProj)
{
	float2 uv = (posProj / posProj.w).xy;
	uv = float2(0.5f, -0.5f) * uv + 0.5f; // Projection space to UV space transform
	return uv;
}

float4 GetLightUV(float4 posWorld, float4x4 viewProjection)
{
	float4 luv = mul(posWorld, viewProjection);
	//luv = luv / luv.w; // Normalize homogenous coords
	//luv.xy = float2(0.5f, -0.5f) * luv.xy + 0.5f; // Projection space to UV space transform
	return luv;
}

#define SkinVertex \
	float4x3 skinning = 0; \
	[unroll] \
	for (int i = 0; i < boneCount; i++) \
		skinning += Bones[vin.Indices[i]] * vin.Weights[i]; \
	vin.Position.xyz = mul(vin.Position, skinning);

#define SkinVertexWithNormal \
	SkinVertex; \
	vin.Normal = mul(vin.Normal, (float3x3)skinning);

#define SkinVertexWithNormalTagent \
	SkinVertexWithNormal;\
	vin.Tangent = mul(vin.Tangent, (float3x3)skinning);

#define SetPosition \
	float4 posWorld = mul(vin.Position, World); \
	vout.pos = mul(posWorld, ViewProjection);

#define SetPositionUV \
	vout.posUV = GetLightUV(posWorld, ViewProjection);

#define SetPositionNormalToEye \
	SetPosition; \
	vout.normal = mul(vin.Normal, (float3x3)World); \
	vout.toEye = posWorld.xyz - EyePosition;

#define SetPositionNormalToEyeTangentBinormal \
	SetPositionNormalToEye; \
	vout.tangent = mul(vin.Tangent, (float3x3)World); \
	vout.binormal = cross(vout.normal,vout.tangent);

#define SetTextureCoord \
	vout.uv = vin.TexCoord;

#define SetLightUVOne \
	vout.lightUv[0] = GetLightUV(posWorld, LightViewProjection[0]);

//#define SetLightUVTwo \ 
//	vout.lightUv0 = GetLightUV(posWorld, LightViewProjection[0]); \
//	vout.lightUv1 = GetLightUV(posWorld, LightViewProjection[1]);
//
//
//#define SetLightUVThree \ 
//	vout.lightUv0 = GetLightUV(posWorld, LightViewProjection[0]); \
//	vout.lightUv1 = GetLightUV(posWorld, LightViewProjection[1]); \
//	vout.lightUv2 = GetLightUV(posWorld, LightViewProjection[2]);
//
//
//#define SetLightUVTwo \ 
//	vout.lightUv0 = GetLightUV(posWorld, LightViewProjection[0]); \
//	vout.lightUv1 = GetLightUV(posWorld, LightViewProjection[1]); \
//	vout.lightUv2 = GetLightUV(posWorld, LightViewProjection[2]); \
//	vout.lightUv3 = GetLightUV(posWorld, LightViewProjection[3]);