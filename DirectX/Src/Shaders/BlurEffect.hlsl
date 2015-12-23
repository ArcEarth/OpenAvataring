#define MaxSampleCount 15
static const float  brightThreshold = 0.35f;
static const float  glowScale = 1.0f;

SamplerState pointSampler : register (s0);
SamplerState linearSampler : register (s1);
Texture2D s0 : register(t0);
Texture2D s1 : register(t1);

cbuffer cb0
{
	float2 sampleOffsets[MaxSampleCount];
	float4 sampleWeights[MaxSampleCount];
	uint   sampleCount;
	float  pixelWidth;
	float  pixelHeight;
	float  multiplier;
};

struct QuadVertexShaderOutput
{
    float4 pos : SV_POSITION;              
    float2 tex : TEXCOORD0;
};

float4 DownScale3x3Ex(QuadVertexShaderOutput input) : SV_TARGET
{
    float4 brightColor = 0.0f;

    // Gather 16 adjacent pixels (each bilinear sample reads a 2x2 region)
    brightColor  = s0.Sample(linearSampler, input.tex + float2(-pixelWidth,-pixelHeight));
    brightColor += s0.Sample(linearSampler, input.tex + float2( pixelWidth,-pixelHeight));
    brightColor += s0.Sample(linearSampler, input.tex + float2(-pixelWidth, pixelHeight));
    brightColor += s0.Sample(linearSampler, input.tex + float2( pixelWidth, pixelHeight));
    brightColor /= 4.0f;

    // Brightness thresholding
    //brightColor = max(0.0f, brightColor - brightThreshold);

    //return float4(brightColor, 1.0f);
    return brightColor;
}

float4 DownScale3x3(QuadVertexShaderOutput input) : SV_TARGET
{   
    float4 brightColor = 0.0f;

    // Gather 16 adjacent pixels (each bilinear sample reads a 2x2 region)
    brightColor  = s0.Sample(linearSampler, input.tex, int2(-1,-1));
    brightColor += s0.Sample(linearSampler, input.tex, int2( 1,-1));
    brightColor += s0.Sample(linearSampler, input.tex, int2(-1, 1));
    brightColor += s0.Sample(linearSampler, input.tex, int2( 1, 1));
    brightColor /= 4.0f;

    // Brightness thresholding
    //brightColor = max(0.0f, brightColor - brightThreshold);

    return brightColor;

 //   return float4(brightColor, 1.0f);
}

float4 UpScale(QuadVertexShaderOutput input) : SV_TARGET
{
	return s0.Sample(linearSampler, input.tex);
}

struct PSOutput
{
	float4 Color : SV_Target;
	float  Depth : SV_Depth;
};

static const float bias = 3e-3f;
PSOutput AlphaAsDepthPassBy(QuadVertexShaderOutput input)
{
	PSOutput pixel;
	float4 value = s0.Sample(linearSampler, input.tex);
	pixel.Color = float4(value.rgb * multiplier, value.g * multiplier);
	pixel.Depth = value.a + bias;
    return pixel;
};

float4 PassBy(QuadVertexShaderOutput input) : SV_TARGET
{
	float4 color = s0.Sample(linearSampler, input.tex);
	return color;
};

float4 Combination(QuadVertexShaderOutput input) : SV_TARGET
{   

    float4 orginal = s0.Sample(pointSampler, input.tex);         // sample scene image from intermediate texture

    float4 blured = s1.Sample(linearSampler, input.tex);     // sample from final glow image    

    float4 final = blured;
    final.w = orginal.r < 0.5f ? orginal.w : blured.w; 

    return final;
}

float4 Blur(QuadVertexShaderOutput input) : SV_TARGET
{    
    float4 glow = 0.0f;
    float4 color = 0.0f;
    float2 samplePosition;

    for(uint sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        // Sample from adjacent points
        samplePosition = input.tex + sampleOffsets[sampleIndex];
        color = s0.Sample(pointSampler, samplePosition);

        glow += sampleWeights[sampleIndex] * color;
    }

    return glow;
}
