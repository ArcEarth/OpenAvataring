#ifndef __GPU
#define __GPU 
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
    result.xyz = -q.
xyz;
    result.w = q.
w;
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
    result *= float4(axis, 1.0f);
    return
result;
}

float3 quat_rotate(float3 v, float4 q) __GPU
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

float3 udq_translation(float2x4 dual) __GPU
{
    return 2.0 * (dual[0].w * dual[1].xyz - dual[1].w * dual[0].xyz + cross(dual[0].xyz, dual[1].xyz));
}

void dq_normalize(inout float2x4 dual) __GPU
{
    float l = length(dual[0]);
    dual /= dual;
}

void udq_transform(float2x4 dual, inout float3 position, inout float3 normal, inout float3 tangent) __GPU
{
    float3 pos = quat_rotate(position, dual[0]);
    float3 translation = 2.0 * (dual[0].w * dual[1].xyz - dual[1].w * dual[0].xyz + cross(dual[0].xyz, dual[1].xyz));
    pos += translation;
    
    position = pos;
    normal = quat_rotate(normal, dual[0]);
    tangent = quat_rotate(tangent, dual[0]);
}
