
cbuffer QuadVertexShaderConstant : register(b0)
{
	float2 uv_base;
	float2 uv_range;
};

struct QuadVertexShaderInput
{
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

struct QuadVertexShaderOutput
{
    float4 pos : SV_POSITION;              
    float2 tex : TEXCOORD0;
};

QuadVertexShaderOutput QuadVertexShader(QuadVertexShaderInput input)
{
    QuadVertexShaderOutput output;
    output.pos = input.pos;
    output.tex = uv_base + input.tex * uv_range;
    return output;
}