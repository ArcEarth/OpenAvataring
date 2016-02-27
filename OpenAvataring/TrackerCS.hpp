#ifndef __GPU // HLSL shader case
#define __FXC
Buffer<float4>  animation;
Buffer<float4>  samples;
RWBuffer<float> likilihoods;

#ifndef __CBUFFER
#define __CBUFFER(name) cbuffer name
#endif

#include "Quaternion.hlsli"
#include "TrackerCS_cbuffer.hlsli"
#endif

float4 GetBoneRelativeRotationAtFrame(uint fid, uint bid) __GPU
{
    return animation[fid * constants.numBones + bid];
}

float4 GetBoneRelativeRotationAtTime(float time, uint bid) __GPU
{
    uint fid = trunc(time / constants.timeSlice);
    float t = fmod(time, constants.timeSlice);
    float4 f0 = GetBoneRelativeRotationAtFrame(fid, bid);
    float4 f1 = GetBoneRelativeRotationAtFrame(fid + 1, bid);
    // lerp the quaternion insetead of slerp
    // since the difference are relative small
    return f0 * (1 - t) + f1 * t; 
}

float4 GetScaledBoneRelativeRotationAtTime(float time, float scale, uint bid) __GPU
{
    float4 q = GetBoneRelativeRotationAtTime(time, bid);
    return quat_scale_rotation(q, scale);
}

float3 GetEffectorFeatureAtTime(float time, float scale, uint bid) __GPU
{
    float3 gt = 0;
    // caculate the vector from chain begin to chain end
    for (int cx = constants.jointsInChain[bid]; cx > 0; --cx)
    {
        float4 q0 = constants.bindPoses[bid][0];
        float3 t = constants.bindPoses[bid][1].xyz;
        // assert(t.w == 0);
        float4 q = GetScaledBoneRelativeRotationAtTime(time, scale, bid);

        q = quat_mul(q,q0);
        t = quat_rotate(t, q);

        gt += t;

        bid = constants.parents[bid];
    }

    // rotate the chain vector to global orientation
    for (int cx = constants.numAncestors[bid] - constants.jointsInChain[bid]; cx > 0; --cx)
    {
        float4 q0 = constants.bindPoses[bid][0];

        float4 q = GetScaledBoneRelativeRotationAtTime(time, scale, bid);

        q = quat_mul(q, q0);
        gt = quat_rotate(gt, q);

        bid = constants.parents[bid];
    }

    return gt;
}

void write_likilihood(uint id) __GPU
{
    float4 particle = samples[id];
    float time = particle.x;
    float scale = particle.y;
    float vt = particle.z;
    float dt = constants.timeDelta * vt;

    float err = 0;

    for (int i = 0; i < constants.numEffectors; i++)
    {
        uint eid = constants.effectors[i];
        float3 fv = GetEffectorFeatureAtTime(time, scale, eid);
        float3 fv0 = GetEffectorFeatureAtTime(time - dt, scale, eid);

        float3 vel = (fv - fv0) / dt;
        vel = stable_normalize(vel);

        float3 spos = constants.effectorPoses[i][0].xyz;
        float3 svel = constants.effectorPoses[i][1].xyz;

        fv = (fv - spos) / constants.poseSigma;
        err += dot(fv, fv);

        vel = (vel - svel) / constants.velSigma;
        err += dot(vel, vel);
    }

    float lik = exp(-err);

    lik *= exp(-sqr(max(abs(scale - constants.uS) - constants.thrS, .0f)) / constants.varS);

    lik *= exp(-sqr(max(abs(vt - constants.thrVt), .0f)) / constants.varVt);

    likilihoods[id] = lik;
}

#ifdef __FXC // HLSL shader case
[numthreads(1, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    uint id = DTid.x;
    write_likilihood(id);
}
#endif