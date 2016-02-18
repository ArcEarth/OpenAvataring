#define __GPU 

Buffer<float4> animation;
Buffer<float4> samples;

#include "TrackerCS_cbuffer.hlsli"

#include "Quaternion.hlsli"

float4 GetBoneRelativeRotationAtFrame(uint fid, uint bid) _GPU
{
    return animation[fid * numBones + bid];
}

float4 GetBoneRelativeRotationAtTime(float time, uint bid) _GPU
{
    uint fid = trunc(time / timeSlice);
    float t = fmode(time, timeSlice);
    float4 f0 = GetBoneRelativeRotationAtFrame(fid, bid);
    float4 f1 = GetBoneRelativeRotationAtFrame(fid + 1, bid);
    // lerp the quaternion insetead of slerp
    // since the difference are relative small
    return f0 * (1 - t) + f1 * t; 
}

float4 quat_scale_rotation(float4 q, float s) _GPU
{
    float4 ag = quat_to_axis_angle(q);
    return quat_from_axis_angle(ag.xyz, ag.w * s);
}

float GetScaledBoneRelativeRotationAtTime(float time, float scale, uint bid) _GPU
{
    float4 q = GetBoneRelativeRotationAtTime(time, bid);
    return quat_scale_rotation(q, scale);
}

float3 GetEffectorFeatureAtTime(float time, float scale, uint bid) _GPU
{
    float3 gt = 0;
    // caculate the vector from chain begin to chain end
    for (int cx = jointsInChain[bid]; cx > 0; --cx)
    {
        float4 q0 = bindPoses[bid][0];
        float3 t = bindPoses[bid][1].xyz; 
        // assert(t.w == 0);
        float4 q = GetScaledBoneRelativeRotationAtTime(time, scale, bid);

        q = quat_mul(q,q0);
        t = quat_rotate(t, q);

        gt += t;

        bid = parents[bid];
    }

    // rotate the chain vector to global orientation
    for (int cx = numAncestor[bid] - jointsInChain[bid]; cx > 0; --cx)
    {
        float4 q0 = bindPoses[bid][0];

        float4 q = GetScaledBoneRelativeRotationAtTime(time, scale, bid);

        q = quat_mul(q, q0);
        gt = quat_rotate(gt, q);

        bid = parents[bid];
    }

    return gt;
}

float3 stable_normalize(float v, uniform float epsilon) _GPU
{
    float l = length(v);
    if (l < epsilon)
        return 0;
    else
        return v / l;
}

[numthreads(1, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    uint id = DTid.x;
    float4 particle = samples[id];
    float time = particle.x;
    float scale = particle.y;
    float vt = particle.z;
    float dt = timeDelta * vt;

    float err = 0;

    for (int i = 0; i < numEffectors; i++)
    {
        uint eid = effectors[i];
        float3 fv = GetEffectorFeatureAtTime(time, scale, eid);
        float3 fv0 = GetEffectorFeatureAtTime(time - dt, scale, eid);

        float3 vel = (fv - fv0) / dt;
        vel = stable_normalize(vel);

        float3 spos = effectorPoses[i][0].xyz;
        float3 svel = effectorPoses[i][1].xyz;

        fv = (fv - spos) / poseSigma;
        err += dot(fv, fv);

        vel = (vel - svel) / velSigma;
        err += dot(vel, vel);
    }

    float posLik = exp(-err);


}