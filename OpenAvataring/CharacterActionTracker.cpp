#include "pch.h"
#include "CharacterActionTracker.h"
#include <Causality\Settings.h>
#include "ArmatureTransforms.h"
#include <type_traits>
#include <random>

#include <amp.h>
#include <amp_graphics.h>
#include <amp_math.h>

#include <amp_hlsl_intrinsic.h>
#include <amp_tinymt_rng.h>

using TrackerScalarType = Causality::IGestureTracker::ScalarType;
extern std::random_device g_rand;
extern std::mt19937 g_rand_mt;
static std::normal_distribution<TrackerScalarType> g_normal_dist(0, 1);

namespace Causality
{
	namespace Internal {
		using namespace concurrency::hlsl;
		using concurrency::graphics::texture_view;
		using concurrency::array_view;
		using concurrency::extent;
		using concurrency::index;
		using concurrency::graphics::texture;

#define uniform 
#define __Unroll__ 

#include "Quaternion.hlsli"

		struct CharacterActionTrackerGpuImpl
		{
		public:
			const static int MaxEndEffectorCount = 32;
			using ScalarType = TrackerScalarType;
			using particle_vector_type = float4; 			//std::conditional_t<std::is_same<ScalarType, float>::value,float4,double4>;
			using likilihood_type = double;

			using anim_texture_t = texture<const particle_vector_type, 2>;

			// Readonly texture that stores the animation data, (Frames X Bones) Dimension
			texture_view<const particle_vector_type, 2>	 animation;	// animation matrix array
			// UAV R/W
			array_view<const particle_vector_type, 1>	 samples;// samples
			// UAV for write
			array_view<likilihood_type, 1>				 likilihoods; // out, likihoods
			//concurrency::array_view<const int, 1>					 parentsView;	// parents
			//concurrency::array_view<const particle_vector_type, 1>	 bindView;

			// this struct is mapped to an constant buffer in DirectX shader
#define __CBUFFER(name) struct name##_t
#include "TrackerCS_cbuffer.hlsli"
			constants_t constants;

#include "TrackerCS.hpp"
#undef uniform
#undef __Unroll__

			void likilihood(concurrency::index<1> idx) __GPU_ONLY
			{
				auto id = idx[0];
				float4 particle = samples[id];

				// get random number in gpu is not that easy, lets do that in cpu
				// particle = progate(particle);

				write_likilihood(idx[0], particle);
			}

			// CPU only excution parts
			// Sample matrix should be stores in row major for efficent access
			using AnimationBuffer = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
			using SampleBuffer = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
			using LiksBuffer = Eigen::VectorXd;

			//concurrency::accelerator_view acc_view;

			// Setup the buffers
			CharacterActionTrackerGpuImpl(const concurrency::accelerator_view& acc_view, const anim_texture_t &anim_texture, const AnimationBuffer& animBuffer, double animationDuration, const SampleBuffer& esamples, LiksBuffer& eliks) __CPU_ONLY
				:
				animation(anim_texture),
				samples(concurrency::extent<1>(esamples.rows()), reinterpret_cast<const particle_vector_type*>(esamples.data())),
				likilihoods(concurrency::extent<1>(eliks.size()), eliks.data())
			{
				auto numFrames = animBuffer.rows();
				constants.timeSlice = (float)(animationDuration / (double)numFrames);
			}

			void setup_effectors(const Causality::ShrinkedArmature& parts, gsl::span<const Causality::ArmaturePart*> effectors) __CPU_ONLY
			{
				constants.numEffectors = effectors.size();
				for (int i = 0; i < effectors.size(); i++)
				{
					auto& part = *effectors[i];
					constants.effectors[i] = effectors[i]->Index;
					for (int j = 0; j < part.Joints.size(); j++)
					{
						auto& joint = *part.Joints[j];
						int jid = joint.ID;
						constants.jointsInChain[jid] = j + 1;
					}
				}


				auto& armature = parts.Armature();
				auto& dframe = armature.bind_frame();
				constants.numBones = armature.size();
				for (int i = 0; i < armature.size(); i++)
				{
					auto& joint = *armature[i];
					int jid = joint.ID;
					constants.parents[jid] = joint.ParentID;
					constants.bindPoses[jid][0] = reinterpret_cast<const float4&>(dframe[jid].LclRotation);
					constants.bindPoses[jid][1] = reinterpret_cast<const float4&>(dframe[jid].LclTranslation);
					if (joint.ParentID == -1)
						constants.numAncestors[jid] = 0;
					else
						constants.numAncestors[jid] = constants.numAncestors[joint.ParentID] + 1;
				}


				constants.uS = 1.0f;
				constants.thrS = 0.2f;
				constants.varS = 0.2f;

				constants.uVt = 1.0f;
				constants.varVt = 0.3f;
				constants.thrVt = 1.3f;

				constants.poseSigma = 1.0f;
				constants.velSigma = 1.0f;
			}

			~CharacterActionTrackerGpuImpl() __CPU_ONLY
			{}

			static anim_texture_t create_animation_texture(const AnimationBuffer& animBuffer, const concurrency::accelerator_view& acc_view)
			{
				anim_texture_t texture(concurrency::extent<2>(animBuffer.rows(), animBuffer.cols() / 4), //dimension
					reinterpret_cast<const float_4*>(animBuffer.data()), // begin
					reinterpret_cast<const float_4*>(animBuffer.data() + animBuffer.size()), // end
					acc_view);
				return texture;
			}

			// The result will be write back into the eigen matrix
			void step_likilihoods(float time_delta) __CPU_ONLY
			{
				constants.timeDelta = time_delta;
			}
		};
	}
}

using namespace Causality;
using namespace Eigen;

CharacterActionTracker::CharacterActionTracker(const ArmatureFrameAnimation & animation, const PartilizedTransformer &transfomer)
	: m_Animation(animation),
	m_Transformer(transfomer),
	m_confidentThre(0.00001),
	m_uS(1.0),
	m_thrVt(1.3),
	m_thrS(0.2),
	m_stepSubdiv(1),
	m_tSubdiv(75),
	m_scaleSubdiv(3),
	m_vtSubdiv(1),
	m_currentValiad(false)
{}

CharacterActionTracker::~CharacterActionTracker()
{

}

CharacterActionTracker::CharacterActionTracker(const CharacterActionTracker & rhs)
	: m_Animation(rhs.m_Animation),
	m_Transformer(rhs.m_Transformer),
	m_confidentThre(rhs.m_confidentThre),
	m_uS(rhs.m_uS),
	m_thrVt(rhs.m_thrVt),
	m_thrS(rhs.m_thrS),
	m_stepSubdiv(rhs.m_stepSubdiv),
	m_tSubdiv(rhs.m_tSubdiv),
	m_scaleSubdiv(rhs.m_scaleSubdiv),
	m_vtSubdiv(rhs.m_vtSubdiv),
	m_currentValiad(rhs.m_currentValiad),
	m_gpu(nullptr)
{
}

void CharacterActionTracker::Reset(const InputVectorType & input)
{
	Reset();
	SetInputState(input, 1.0f / 30.0f);
	StepParticals();
	m_currentValiad = false;
}

void CharacterActionTracker::Reset()
{
	auto& frames = m_Animation.GetFrameBuffer();

	int tchuck = m_tSubdiv, schunck = m_scaleSubdiv, vchunck = m_vtSubdiv;

	m_dt = m_Animation.Duration.count() / tchuck;
	auto dt = m_dt;

	Eigen::Matrix<TrackerScalarType,1,3> v;
	auto numParticles = tchuck * schunck * vchunck;
	m_sample.resize(numParticles, 3);
	m_liks.resize(numParticles);
	m_newSample.resize(numParticles, 3);
	m_newLiks.resize(numParticles);

	auto stdevS = sqrt(m_varS);
	auto stdevV = sqrt(m_varVt);

	//v[3] = .0f;
	for (int i = 0; i < tchuck; i++)
	{
		v[0] = i * dt;
		float s = m_uS; // - stdevS
		if (tchuck > 1)
			s -= stdevS;
		for (int j = 0; j < schunck; j++, s += 2 * stdevS / (schunck - 1))
		{
			v[1] = s;

			float vt = .0f;//-stdevV;
			if (vchunck > 1)
				vt -= stdevV;
			for (int k = 0; k < vchunck; k++, vt += 2 * stdevV / (vchunck - 1))
			{
				v[2] = vt;
				m_sample.row(i * schunck * vchunck + j * vchunck + k) = v;
			}
		}
	}

	m_liks.array() = 1.0f / (double)numParticles;

	m_fvectors.resize(m_sample.rows(), m_CurrentInput.cols());

	m_currentValiad = false;
}

CharacterActionTracker::ScalarType CharacterActionTracker::Step(const InputVectorType & input, ScalarType dt)
{
	double confi = 0;

	int steps = m_currentValiad ? m_stepSubdiv : 1;
	InputVectorType previnput = m_currentValiad ? m_CurrentInput : input;

	ScalarType tick = ScalarType(1.0) / ScalarType(steps);
	ScalarType tock = 0;
	for (int iter = 0; iter < steps; iter++)
	{
		tock += tick;
		m_CurrentInput = previnput * (ScalarType(1.0) - tock) + input;

		SetInputState(m_CurrentInput, tick * dt);

		confi = StepParticals();
	}

	//m_fvectors.rowwise() -= m_CurrentInput;

	if (confi < m_confidentThre)
	{
		std::cout << "[Tracker] *Rest*************************" << std::endl;
		Reset(input);
		confi = m_liks.sum();
	}
	return confi;
}

void CharacterActionTracker::SetInputState(const InputVectorType & input, ScalarType dt)
{
	m_CurrentInput = input;
	m_currentValiad = true;
	m_dt = dt;
	m_lidxCount = 0;
	if (m_fvectors.cols() != input.cols())
		m_fvectors.resize(NoChange, input.cols());
}

inline float sqr(float x) __CPU_ONLY
{
	return x * x;
}

inline double sqr(double x) __CPU_ONLY
{
	return x * x;
}

void CharacterActionTracker::Progate(TrackingVectorBlockType & x)
{
	auto& t = x[0];
	auto& s = x[1];
	auto& vt = x[2];
	//auto& ds = x[3];

	vt += g_normal_dist(g_rand_mt) * m_stdevDVt * m_dt;
	auto dt = vt * m_dt;
	// ds += ;

	t += dt;
	t = fmod(t, m_Animation.Duration.count());
	if (t < 0)
		t += m_Animation.Duration.count();

	s += g_normal_dist(g_rand_mt) * m_stdevDs * m_dt;
}


CharacterActionTracker::LikilihoodScalarType CharacterActionTracker::Likilihood(int idx, const TrackingVectorBlockType & x)
{
	using namespace std;

	InputVectorType vx;
	vx = GetCorrespondVector(x, ArmatureFrameView(s_frameCache0), ArmatureFrameView(s_frameCache1));
	//m_fvectors.row(m_lidxCount++) = vx;

	// Distance to observation
	using scalar = LikilihoodScalarType;

	InputVectorType diff = (vx - m_CurrentInput).cwiseAbs2().eval();
	LikilihoodScalarType likilihood = (diff.array() / m_LikCov.array()).sum();
	likilihood = exp(-likilihood);

	// Scale factor distribution
	likilihood *= exp(-sqr(max((scalar)(abs(scalar(x[1]) - scalar(m_uS)) - scalar(m_thrS)), scalar(.0))) / scalar(m_varS));
	// Speed scale distribution
	likilihood *= exp(-sqr(max((scalar)abs(scalar(x[2]) - scalar(m_thrVt)), scalar(.0f))) / scalar(m_varVt));

	//return 1.0;
	return likilihood;
}

CharacterActionTracker::InputVectorType CharacterActionTracker::GetCorrespondVector(const TrackingVectorBlockType & x, ArmatureFrameView frameCache0, ArmatureFrameView frameChache1) const
{
	InputVectorType vout;
	int has_vel = g_TrackerUseVelocity ? (g_NormalizeVelocity ? 2 : 1) : 0;
	//if (x.size() == 3)
	//has_vel = true;

	float t = x[0], s = x[1];

	P2PTransform ctrl;
	ctrl.SrcIdx = PvInputTypeEnum::ActiveParts;
	ctrl.DstIdx = PvInputTypeEnum::ActiveParts;

	GetScaledFrame(frameCache0, t, s);

	if (!has_vel)
	{
		vout = m_Transformer.GetCharacterInputVector(ctrl, frameCache0, frameCache0, .0f, has_vel).cast<ScalarType>();
	}
	else
	{
		float dt = x[2] * m_dt;
		GetScaledFrame(frameChache1, t - dt, s);

		vout = m_Transformer.GetCharacterInputVector(ctrl, frameCache0, frameChache1, dt, has_vel).cast<ScalarType>();
	}
	return vout;
}

void CharacterActionTracker::GetScaledFrame(_Out_ ArmatureFrameView frame, ScalarType t, ScalarType s) const
{
	m_Animation.GetFrameAt(frame, time_seconds(t), false);
	FrameScaleEst(frame, m_Animation.DefaultFrame, s);
	FrameRebuildGlobal(m_Animation.Armature(), frame);
}

void CharacterActionTracker::SetLikihoodVarience(const InputVectorType & v)
{
	m_LikCov = v;
}

void CharacterActionTracker::SetTrackingParameters(ScalarType stdevDVt, ScalarType varVt, ScalarType stdevDs, ScalarType varS)
{
	m_stdevDVt = stdevDVt;
	m_varVt = varVt;
	m_stdevDs = stdevDs;
	m_varS = varS;
}

void CharacterActionTracker::SetVelocityTolerance(ScalarType thrVt)
{
	m_thrVt = thrVt;
}

void CharacterActionTracker::SetScaleTolerance(ScalarType thrS)
{
	m_thrS = thrS;
}

void CharacterActionTracker::SetStepSubdivition(int subdiv) {
	m_stepSubdiv = subdiv;
}

void CharacterActionTracker::SetParticalesSubdiv(int timeSubdiv, int scaleSubdiv, int vtSubdiv)
{
	m_tSubdiv = timeSubdiv;

	if (!(vtSubdiv & 1))
		vtSubdiv += 1;
	m_vtSubdiv = vtSubdiv;

	if (!(scaleSubdiv & 1))
		scaleSubdiv += 1;
	m_scaleSubdiv = scaleSubdiv;
}

thread_local Bone CharacterActionTracker::s_frameCache0[FrameCacheSize];
thread_local Bone CharacterActionTracker::s_frameCache1[FrameCacheSize];
