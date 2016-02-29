#ifndef __GPU
// the case for HLSL shader source
#define __GPU 
#define __GPU_ONLY
#define __Inout_Param__(T,name) inout T name
#define __Out_Param__(T,name) out T name
#define __Unroll__ [unroll]
#else
#endif
// returns the quaternion of q2*q1
// which represent the rotation sequence of q1 followed by q2
float4 quat_mul(float4 q1, float4 q2) __GPU
{
float4 result;
    result.x = (q2.w * q1.x) + (q2.x * q1.w) + (q2.y * q1.z) - (q2.z * q1.y);
    result.y = (q2.w * q1.y) - (q2.x * q1.z) + (q2.y * q1.w) + (q2.z * q1.x);
    result.z = (q2.w * q1.z) + (q2.x * q1.y) - (q2.y * q1.x) + (q2.z * q1.w);
    result.w = (q2.w * q1.w) - (q2.x * q1.x) - (q2.y * q1.y) - (q2.z * q1.z);
    return
result;
}

float4 quat_conj(float4 q) __GPU
{
float4 result;
    result.xyz = -q.xyz;
    result.w = q.w;
    return
result;
}

float4 quat_to_axis_angle(float4 q) __GPU
{
float4 result;
    result.xyz = normalize(q.xyz);
    result.w = 2.0f * acos(q.w);
    return
result;
}

float4 quat_from_axis_angle(float3 axis, float angle) __GPU
{
    float s, c;
    sincos(0.5f * angle, s, c);
    float4 result = float4(s, s, s, c);
    result.xyz *= axis;
    return result;
}

float3 quat_rotate(float3 v, float4 q) __GPU
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

float3 udq_translation(float2x4 dual) __GPU
{
    return 2.0 * (dual[0].w * dual[1].xyz - dual[1].w * dual[0].xyz + cross(dual[0].xyz, dual[1].xyz));
}

void dq_normalize(__Inout_Param__(float2x4,dual)) __GPU
{
    float l = length(dual[0]);
    dual /= l;
}

void udq_transform(float2x4 dual,
 __Inout_Param__(float3,position),
 __Inout_Param__(float3,normal),
 __Inout_Param__(float3,tangent)) __GPU
{
    position = quat_rotate(position, dual[0]) + udq_translation(dual);
    normal = quat_rotate(normal, dual[0]);
    tangent = quat_rotate(tangent, dual[0]);
}

float4 quat_scale_rotation(float4 q, float s) __GPU
{
    float4 ag = quat_to_axis_angle(q);
    return quat_from_axis_angle(ag.xyz, ag.w * s);
}


float3 stable_normalize(float3 v) __GPU
{
    float l = length(v);
    if (l < 0.001)
        return v;
    else
        return v / l;
}

float sqr(float x) __GPU
{
    return x * x;
}

// nvidia GPU should be scalar FPU, thats no vectorization are needed
float4 slerp(float4 q0, float4 q1, float t) __GPU
{
    // Precomputed constants.
    uniform const float opmu = 1.90110745351730037f;
    uniform const float u[8] = // 1 /[i (2i + 1 )] for i >= 1
    {
    1.f/(1 * 3), 1.f/(2 * 5), 1.f/(3 * 7), 1.f/(4 * 9),
    1.f/(5 * 11), 1.f/(6 * 13), 1.f/(7 * 15), opmu/(8 * 17)
    } ;
    uniform const float v[8] = // i /(2 i+ 1) for i >= 1
    {
    1.f/3, 2.f/5, 3.f/7, 4.f/9,
    5.f/11, 6.f/13, 7.f/15, opmu * 8/17
    } ;

    float x = dot(q0,q1); // cos (theta)
    float signx = (x >= 0 ? 1 : (x = -x, -1));
    float xm1 = x - 1;
    float d = 1 - t, sqrT = t * t, sqrD = d * d;
    float bT[8], bD[8];
    __Unroll__
    for (int i = 7; i >= 0; --i)
    {
    bT[i] = (u[i] * sqrT - v[i]) * xm1;
    bD[i] = (u[i] * sqrD - v[i]) * xm1;
    }
    float cT = signx * t *(
    1 + bT[0] * (1 + bT[1] * (1 + bT[2] * (1 + bT[3] * (
    1 + bT[4] * (1 + bT[5] * (1 + bT[6] * (1 + bT [7] ))))))));
    float cD = d * (
    1 + bD[0] * (1 + bD[1] * (1 + bD[2] * (1 + bD[3] * (
    1 + bD[4] * (1 + bD[5] * (1 + bD[6] * (1 + bD[7] ))))))));
    return q0 * cD + q1 * cT;
}