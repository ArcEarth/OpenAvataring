#define MAX_BONES 64
#define MAX_EFFECTORS 16

__CBUFFER(constants)
{
    uint numBones;
    uint numEffectors;
    uint _padding0[2];

    // effector info
    uint effectors[MAX_EFFECTORS];
    float2x4
         effectorPoses[MAX_EFFECTORS];

    // skeleton hierachy info
    uint parents[MAX_BONES];
    uint numAncestors[MAX_BONES];
    uint jointsInChain[MAX_BONES];
    float2x4 bindPoses[MAX_BONES];

    // runtime info
    float timeSlice;
    float timeDelta;
    float _padding1[2];

    // tracker parameters
    float3 poseSigma;
    float _padding2;
    float3 velSigma;
    float _padding3;

    float uS;
    float thrS;
    float varS;
    float _padding4;

    float uVt;
    float thrVt;
    float varVt;
    float _padding5;
};
