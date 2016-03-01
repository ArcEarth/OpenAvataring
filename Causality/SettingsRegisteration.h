SETTING_REGISTERATION(int, PartAssignmentTransform, PAT_RST)
SETTING_REGISTERATION(int, HandleTransformAnisometricMin, 0.2)
SETTING_REGISTERATION(int, HandleTransformAnisometricMax, 1.2)
SETTING_REGISTERATION(bool, UsePersudoPhysicsWalk, false)
SETTING_REGISTERATION(int, PhaseMatchingInterval, 1)

SETTING_REGISTERATION(float, MaxCharacterSpeed, 0.5)
SETTING_REGISTERATION(int, EnableDebugLogging, 1)
SETTING_REGISTERATION(bool, EnableRecordLogging, false)

SETTING_REGISTERATION(bool, StepFrame, false)

//	When this flag set to true, subtle motion will be synthesised by major motion
SETTING_REGISTERATION(bool, EnableDependentControl, false)

//	When this flag set to true, the kinect input's rotation will be ingnored
SETTING_REGISTERATION(bool, IngnoreInputRootRotation, true)

SETTING_REGISTERATION(float, FrameTimeScaleFactor, 20)

SETTING_REGISTERATION(float, PlayerPcaCutoff, 0.04f) // 0.2^2
//static const float	EnergyCutoff = 0.3f;

SETTING_REGISTERATION(float, PlayerActiveEnergy, 0.3f)
SETTING_REGISTERATION(float, PlayerSubactiveEnergy, 0.03f)

SETTING_REGISTERATION(float, BlendWeight, 0.8f)

SETTING_REGISTERATION(float, AssignWeight, 1.0f)

SETTING_REGISTERATION(float, MatchAccepetanceThreshold, 0.2f)

SETTING_REGISTERATION(bool, EnableInputFeatureLocalization, true)

SETTING_REGISTERATION(bool, LoadCharacterModelParameter, true)
SETTING_REGISTERATION(float, CharacterPcaCutoff, 0.004f) // 0.2^2
SETTING_REGISTERATION(float, CharacterActiveEnergy, 0.40f)
SETTING_REGISTERATION(float, CharacterSubactiveEnergy, 0.02f)

SETTING_REGISTERATION(float, IKTermWeight, 1.0f)
SETTING_REGISTERATION(float, MarkovTermWeight, 1.0f)
SETTING_REGISTERATION(float, StyleLikelihoodTermWeight, 1.0f)
SETTING_REGISTERATION(float, IKLimitWeight, 1.0)

SETTING_REGISTERATION(float, DebugArmatureThinkness, 0.005f)

SETTING_REGISTERATION(bool, DebugView, false)

SETTING_REGISTERATION(bool, ShowCharacterMesh, true)

SETTING_REGISTERATION(bool, MirrowInputX, false)

SETTING_REGISTERATION(bool, UseJointLengthWeight, false)

SETTING_REGISTERATION(bool, UseStylizedIK, true)
SETTING_REGISTERATION(bool, UseVelocity, false)
SETTING_REGISTERATION(bool, NormalizeVelocity, false)
SETTING_REGISTERATION(float, VelocityNormalizeThreshold, 0.01)


SETTING_REGISTERATION(bool, ForceRemappingAlwaysOn, false)
SETTING_REGISTERATION(float, RevampLikilyhoodThreshold, 0.05f)
SETTING_REGISTERATION(float, RevampLikilyhoodTimeThreshold, 1.0f)
SETTING_REGISTERATION(float, RevampActiveSupportThreshold, 0.65f)

SETTING_REGISTERATION(float, StructrualSymtricBonus, 0.2f)
SETTING_REGISTERATION(float, StructrualDisSymtricPenalty, -0.2f)
SETTING_REGISTERATION(float, MaxiumTimeDelta, 1.0 / 30.0)

SETTING_REGISTERATION(float, DynamicTraderKeyEnergy, 0.2f)
SETTING_REGISTERATION(double, DynamicTraderCurvePower, 1.0)
SETTING_REGISTERATION(double, DynamicTraderSpeedFilterCutoffFrequency, 1.0)
SETTING_REGISTERATION(double, CharacterJointFilterCutoffFrequency, 1.0)

SETTING_REGISTERATION(float, DefaultTrackerCovierence, 1000000000)
SETTING_REGISTERATION(double, TrackerNormalizedVelocityVariance, 0.517*0.517)

SETTING_REGISTERATION(double, TrackerRestConfident, 1e-7)

SETTING_REGISTERATION(bool, TrackerGpuAcceleration, true)
SETTING_REGISTERATION(bool, TrackerUseVelocity, false)

SETTING_REGISTERATION(int, TrackerTopK, 30)
SETTING_REGISTERATION(int, TrackerSubStep, 5)
SETTING_REGISTERATION(int, TrackerVtSubdivide, 1)
SETTING_REGISTERATION(int, TrackerSclSubdivide, 3)
SETTING_REGISTERATION(double, TrackerTimeSubdivide, 1.0)
SETTING_REGISTERATION(double, TrackerStDevVt, 0.3)
SETTING_REGISTERATION(double, TrackerStDevScale, 0.3)
SETTING_REGISTERATION(double, TrackerVtProgationStep, 0.1)
SETTING_REGISTERATION(double, TrackerVtThreshold, 1.5)
SETTING_REGISTERATION(double, TrackerScaleThreshold, 0.15)
SETTING_REGISTERATION(double, TrackerScaleProgationStep, 0.1)
SETTING_REGISTERATION(double, TrackerSwitchCondifidentThreshold, 0.5)
SETTING_REGISTERATION(double, TrackerSwitchTimeThreshold, 1.0)
