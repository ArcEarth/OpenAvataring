#include "pch.h"
#include "CharacterActionTracker.h"
#include <Causality\Settings.h>
#include "ArmatureTransforms.h"
#include <type_traits>
#include <random>
#include <chrono>

#include <amp.h>
#include <amp_graphics.h>
#include <amp_math.h>

#include <amp_hlsl_intrinsic.h>

#ifndef _DEBUG
#define openMP
#endif

using TrackerScalarType = Causality::IGestureTracker::ScalarType;
static std::normal_distribution<TrackerScalarType> g_normal_dist(0, 1);
static std::uniform_real<TrackerScalarType> g_uniform(0, 1);

std::random_device g_rand;
std::mt19937 g_rand_mt(g_rand());

//Causality::wptr<Causality::CharacterActionTracker::GpuAcceleratorView>	Causality::CharacterActionTracker::s_accView;

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

		struct CharacterActionTrackerGpuImplBase
		{
		public:
			const static int MaxEndEffectorCount = 32;
			using ScalarType = TrackerScalarType;
			using particle_vector_type = float3; 			//std::conditional_t<std::is_same<ScalarType, float>::value,float4,double4>;
			using likilihood_type = double;

			using anim_texture_t = texture<float4, 2>;
			using anim_view_t = texture_view<const float4, 2>;
			using sample_view_t = array_view<const particle_vector_type, 1>;
			using likilihood_view_t = array_view<likilihood_type, 1>;

			// Readonly texture that stores the animation data, (Frames X Bones) Dimension
			anim_view_t			animation;	// animation matrix array
			// UAV R/W
			sample_view_t		samples;// samples
			// UAV for write
			likilihood_view_t	likilihoods; // out, likihoods
			//concurrency::array_view<const int, 1>					 parentsView;	// parents
			//concurrency::array_view<const particle_vector_type, 1>	 bindView;

			// this struct is mapped to an constant buffer in DirectX shader
#define __CBUFFER(name) struct name##_t
#include "TrackerCS_cbuffer.inl"
			constants_t constants;

#pragma push_macro("__GPU_ONLY")
#undef __GPU_ONLY
#define __GPU_ONLY const restrict(amp)
#include "TrackerCS.inl"
#undef uniform
#undef __Unroll__

			void operator()(concurrency::index<1> idx) __GPU_ONLY
			{
				auto id = idx[0];
				float3 particle = samples[id];

				//likilihoods[idx] = particle.x;
				// get random number in gpu is not that easy, lets do that in cpu
				// particle = progate(particle);

				write_likilihood(id, particle);
			}

#pragma pop_macro("__GPU_ONLY")

			// CPU only excution parts
			// Sample matrix should be stores in row major for efficent access
			using AnimationBuffer = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
			using SampleBuffer = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
			using LiksBuffer = Eigen::VectorXd;

			//concurrency::accelerator_view acc_view;

			// Setup the buffers
			CharacterActionTrackerGpuImplBase(anim_texture_t &anim_texture, double animationDuration, const SampleBuffer& esamples, LiksBuffer& eliks) __CPU_ONLY
				:
				animation(anim_texture),
				samples(concurrency::extent<1>(esamples.rows()), reinterpret_cast<const particle_vector_type*>(esamples.data())),
				likilihoods(concurrency::extent<1>(eliks.size()), eliks.data())
			{
				ZeroMemory(&constants,sizeof(constants));
				double numFrames = (double)anim_texture.extent[0];
				constants.timeSlice = (float)(animationDuration / numFrames);
			}

			void setup_effectors(const Causality::ShrinkedArmature& parts, gsl::span<const Causality::ArmaturePart*> effectors) __CPU_ONLY
			{
				constants.numEffectors = effectors.size();
				for (int i = 0; i < effectors.size(); i++)
				{
					auto& part = *effectors[i];
					constants.effectors[i] = effectors[i]->Index;
				}

				for (int i = 0; i < parts.size(); i++)
				{
					auto& part = *parts[i];
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

			~CharacterActionTrackerGpuImplBase() __CPU_ONLY
			{}

			// The result will be write back into the eigen matrix
			void step_likilihoods(float time_delta, const concurrency::accelerator_view& acc_view) __CPU_ONLY
			{
				constants.timeDelta = time_delta;
				likilihoods.discard_data();

				//samples.refresh();
				samples.synchronize_to(acc_view);

				concurrency::parallel_for_each(acc_view, likilihoods.extent, *this);

				likilihoods.synchronize(concurrency::access_type_read); //? access_type_read_write?
			}

			// The result will be write back into the eigen matrix
			void step_likilihoods(float time_delta) __CPU_ONLY
			{
				constants.timeDelta = time_delta;
				likilihoods.discard_data();

				samples.refresh();

				concurrency::parallel_for_each(likilihoods.extent, *this);

				likilihoods.synchronize(concurrency::access_type_read); //? access_type_read_write?
			}		
		};

		struct CharacterActionTrackerGpuImpl
		{
			using base_type = CharacterActionTrackerGpuImplBase;

			using anim_texture_t = base_type::anim_texture_t;
			using sample_view_t = base_type::sample_view_t;
			using particle_vector_type = base_type::particle_vector_type;
			using likilihood_view_t = base_type::likilihood_view_t;
			using AnimationBuffer = base_type::AnimationBuffer;
			using SampleBuffer = base_type::SampleBuffer;
			using LiksBuffer = base_type::LiksBuffer;

			//concurrency::accelerator_view av;
			AnimationBuffer				  animBuffer;
			anim_texture_t				  animTexture;
			std::mutex					  acc_mutex;
			base_type					  functor;

			anim_texture_t create_animation_texture(const ArmatureFrameAnimation & animation/*, const concurrency::accelerator_view& acc_view*/) __CPU_ONLY
			{
				const auto& fbuffer = animation.GetFrameBuffer();
				const auto& armature = animation.Armature();
				animBuffer.resize(fbuffer.size(), armature.size() * 4);
				for (int i = 0; i < fbuffer.size(); i++)
				{
					auto& frame = fbuffer[i];
					for (int j = 0; j < armature.size(); j++)
					{
						animBuffer.block<1, 4>(i, j * 4) = reinterpret_cast<const Eigen::RowVector4f&>(frame[j].LclRotation);
					}
				}

				anim_texture_t texture(concurrency::extent<2>(animBuffer.rows(), animBuffer.cols() / 4), //dimension
					reinterpret_cast<const float_4*>(animBuffer.data()), // begin
					reinterpret_cast<const float_4*>(animBuffer.data() + animBuffer.size()) // end
					/*,acc_view*/);
				return texture;
			}

			~CharacterActionTrackerGpuImpl() __CPU_ONLY
			{}

			CharacterActionTrackerGpuImpl(ID3D11Device* pD3dDevice, const ArmatureFrameAnimation & animation, const SampleBuffer& esamples, LiksBuffer& eliks) __CPU_ONLY
				: //av(acc_view),
				animTexture(create_animation_texture(animation /*, acc_view*/)),
				functor(animTexture, animation.Duration.count(), esamples, eliks)
			{
			}

			bool step_likilihoods(float time_delta) __CPU_ONLY
			{
				if (std::try_lock(acc_mutex))
				{
					std::lock_guard<std::mutex> guard(acc_mutex, std::adopt_lock);
					functor.step_likilihoods(time_delta/* ,av*/);
					return true;
				}
				else return false;
			}

			void reset_samples(const SampleBuffer& esamples, LiksBuffer& eliks) __CPU_ONLY
			{
				std::lock_guard<std::mutex> guard(acc_mutex, std::adopt_lock);
				functor.samples = sample_view_t(concurrency::extent<1>(esamples.rows()), reinterpret_cast<const particle_vector_type*>(esamples.data()));
				functor.likilihoods = likilihood_view_t(concurrency::extent<1>(eliks.size()), eliks.data());
			}
		};
	}
}

using namespace Causality;
using namespace Eigen;

struct CharacterActionTracker::GpuAcceleratorView
{
	concurrency::accelerator_view av;
	GpuAcceleratorView(ID3D11Device* pD3dDevice)
		:av(concurrency::direct3d::create_accelerator_view(pD3dDevice))
	{
	}
};


CharacterActionTracker::CharacterActionTracker(const ArmatureFrameAnimation & animation, const PartilizedTransformer &transfomer, IRenderDevice* pDevice)
	: m_Animation(animation),
	m_Transformer(transfomer),
	m_confidentThre(g_TrackerRestConfident),
	m_uS(1.0),
	m_thrVt(1.3),
	m_uVt(.0),
	m_thrS(0.2),
	m_stepSubdiv(1),
	m_tSubdiv(75),
	m_scaleSubdiv(3),
	m_vtSubdiv(1),
	m_currentValiad(false),
	m_parts(m_Transformer.TargetParts()),
	m_effectors(m_Transformer.ActiveParts.size())
{
	auto& aps = m_Transformer.ActiveParts;

	for (size_t i = 0; i < aps.size(); i++)
		m_effectors[i] = m_parts[aps[i].DstIdx];

	m_accelerator = pDevice == nullptr ? SMP : GPU;
	//if (s_accView.expired())
	//{
	//	if (pDevice != nullptr)
	//	{
	//		m_accView = std::make_shared<GpuAcceleratorView>(pDevice);
	//		s_accView = m_accView;
	//	}
	//}
	//else {
	//	m_accView = s_accView.lock();
	//}
}

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
	m_uVt(rhs.m_uVt),
	m_parts(rhs.m_parts),
	m_effectors(rhs.m_effectors),
	//m_accView(rhs.m_accView),
	m_gpu(nullptr)
{
}

void CharacterActionTracker::SwitchAccelerater(AcceceleratorEnum acc)
{
	m_accelerator = acc;
	if (m_gpu)
	{
		if ((void*)m_gpu->functor.samples.data() != (void*)m_sample.data())
		{
			m_gpu->reset_samples(m_sample, m_liks);
		}
	}
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
	m_framesCounter = 0;
	auto& frames = m_Animation.GetFrameBuffer();

	int tchuck = m_tSubdiv, schunck = m_scaleSubdiv, vchunck = m_vtSubdiv;

	m_dt = m_Animation.Duration.count() / tchuck;
	auto dt = m_dt;

	Eigen::Matrix<TrackerScalarType, 1, 3> v;
	auto numParticles = tchuck * schunck * vchunck;

	bool resized = m_sample.rows() != numParticles;
	if (resized)
	{
		m_sample.resize(numParticles, 3);
		m_liks.resize(numParticles);
		m_newSample.resize(numParticles, 3);
		m_newLiks.resize(numParticles);
	}

	m_liks.array() = 1.0f / (double)numParticles;


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

			float vt = m_uVt;//-stdevV;
			if (vchunck > 1)
				vt -= stdevV;
			for (int k = 0; k < vchunck; k++, vt += 2 * stdevV / (vchunck - 1))
			{
				v[2] = vt;
				m_sample.row(i * schunck * vchunck + j * vchunck + k) = v;
			}
		}
	}

	m_fvectors.resize(m_sample.rows(), m_CurrentInput.cols());

	if (m_gpu == nullptr)
	{
		m_gpu.reset(new GpuImpl(m_pDevice, m_Animation, m_sample, m_liks));
		m_gpu->functor.setup_effectors(m_parts, m_effectors);
	}
	else if (resized)
	{
		// Update the GPU view as the CPU data pointer are not valiad anymore
		m_gpu->reset_samples(m_sample, m_liks);
	}

	SetupGpuImplParameters();

	m_currentValiad = false;
}

void CharacterActionTracker::SetupGpuImplParameters()
{
	if (m_gpu)
	{
		auto& constants = m_gpu->functor.constants;

		constants.uS = m_uS;
		constants.thrS = m_thrS;
		constants.varS = m_varS;

		constants.uVt = m_uVt;
		constants.varVt = m_varVt;
		constants.thrVt = m_thrVt;

		assert((m_LikCov.size() % 6 == 0) && "Tracking vector dimension have changed!!!");
		if (m_LikCov.data() == nullptr || m_LikCov.size() == 0)
		{
			constants.poseSigma = 1.0f;
			constants.velSigma = 1.0f;
		}
		else
		{
			constants.poseSigma = *reinterpret_cast<concurrency::graphics::float_3*>(m_LikCov.data());
			constants.velSigma = *reinterpret_cast<concurrency::graphics::float_3*>(m_LikCov.data() + 3);
		}
	}
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
		using namespace  std::chrono;
		auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		std::cout << std::put_time(std::localtime(&now), "[%H:%M:%S]") << " [Tracker:"<< m_framesCounter<<"] Rest at confident (" << confi << ')'<< std::endl;
		Reset(input);
		confi = m_liks.sum();
	}

	++m_framesCounter;
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

	if (m_gpu && m_accelerator == GPU)
	{
		auto& constants = m_gpu->functor.constants;
		auto& eposes = constants.effectorPoses;
		for (int i = 0; i < m_effectors.size(); i++)
		{
			using f3 = concurrency::graphics::float_3;
			eposes[i][0].xyz = *reinterpret_cast<const f3*>(input.segment<3>(i * 6).data());
			eposes[i][1].xyz = *reinterpret_cast<const f3*>(input.segment<3>(i * 6 + 3).data());
		}
	}
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

	// the less velocity, the less acceleration
	// hope this factor could reduce the jitter when almost still
	double vfactor = 1.0;// (std::min(abs(vt) + 0.5, 1.0));
	vt += g_normal_dist(g_rand_mt) * m_VtProgation * m_dt * vfactor;
	auto dt = vt * m_dt;
	// ds += ;

	t += dt;
	t = fmod(t, m_Animation.Duration.count());
	if (t < 0)
		t += m_Animation.Duration.count();

	s += g_normal_dist(g_rand_mt) * m_ScaleProgation * m_dt * vfactor;
	s = fmax(s, .0);
}

double __fastcall platform_gaussian(double x, double mean, double thr, double variance)
{
	x = fmax(x - mean,.0);
	x = fmax(x - thr, .0);
	x = -sqr(x) / variance;
	x = exp(x);
	return x;
}

CharacterActionTracker::LikilihoodScalarType CharacterActionTracker::Likilihood(int idx, const TrackingVectorBlockType & x)
{
	using namespace std;

	InputVectorType vx;
	auto armsize = m_Animation.Armature().size();
	if (s_frameCache0.size() < armsize || s_frameCache1.size() < armsize)
	{
		s_frameCache0.resize(armsize);
		s_frameCache1.resize(armsize);
	}

	vx = GetCorrespondVector(x, s_frameCache0, s_frameCache1);
	//m_fvectors.row(m_lidxCount++) = vx;

	// Distance to observation
	using scalar = LikilihoodScalarType;

	InputVectorType diff = (vx - m_CurrentInput).cwiseAbs2().eval();
	diff.array() /= m_LikCov.array();
	scalar likilihood = diff.sum();
	likilihood = exp(-likilihood);

	// Scale factor distribution
	double scale = x[1];
	double vt = x[2];
	scalar ls = platform_gaussian(scale, m_uS, m_thrS, m_varS);
	// Speed scale distribution
	//! When scale factor are small, high speed does not mean any thing
	//! The space (s,vt) is __NOT_Euclid__, but an Hilbert space
	//! Where the actual distance metric are defined in the mapped Character-Pose space
	scalar lv = platform_gaussian(vt, m_uVt, m_thrVt * scale, m_varVt * vt);

	likilihood *= ls * lv;

	//return 1.0;
	return likilihood;
}

void CharacterActionTracker::Likilihoods_Gpu()
{
	if (m_gpu)
		m_gpu->step_likilihoods(m_dt);
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
	m_VtProgation = stdevDVt;
	m_varVt = varVt;
	m_ScaleProgation = stdevDs;
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

IGestureTracker::~IGestureTracker()
{
}

ParticaleFilterBase::ParticaleFilterBase()
	: m_accelerator(None)
{
}

ParticaleFilterBase::~ParticaleFilterBase()
{
}

ParticaleFilterBase::ScalarType ParticaleFilterBase::Step(const InputVectorType & input, ScalarType dt)
{
	SetInputState(input, dt);
	return StepParticals();
}

const ParticaleFilterBase::TrackingVectorType & ParticaleFilterBase::CurrentState() const
{
	return m_waState;
}

const ParticaleFilterBase::TrackingVectorType & Causality::ParticaleFilterBase::MostLikilyState() const
{
	return m_mleState;
}

int * ParticaleFilterBase::GetTopKStates(int k) const
{
	// instead of max one, we select max-K element
	if (m_srtIdxes.size() < m_sample.rows())
		m_srtIdxes.resize(m_sample.rows());

	std::iota(m_srtIdxes.begin(), m_srtIdxes.end(), 0);
	std::partial_sort(m_srtIdxes.begin(), m_srtIdxes.begin() + k, m_srtIdxes.end(), [this](int i, int j) {
		return m_liks(i) > m_liks(j);
	});
	return m_srtIdxes.data();
}

const ParticaleFilterBase::MatrixType & ParticaleFilterBase::GetSampleMatrix() const { return m_sample; }

const ParticaleFilterBase::LikihoodsType & ParticaleFilterBase::GetSampleLikilihoods() const
{
	return m_liks;
}

void ParticaleFilterBase::Likilihoods_Gpu()
{
	throw new std::runtime_error("this tracker does not provide gpu acceleration methods");
}

ParticaleFilterBase::ScalarType ParticaleFilterBase::StepParticals()
{
	Resample(m_newLiks, m_newSample, m_liks, m_sample);
	if (m_accelerator == AcceceleratorEnum::GPU)
		m_sample = m_newSample; // copy the data, as the GPU memory are mapped to this chuck of CPU memory
	else
		//? This could introduce potential crash when hot-switch from CPU to GPU !!!
		m_newSample.swap(m_sample); // CPU pass, then we can double buffering

	//m_newSample.swap(m_sample);
	//m_newLiks.swap(m_liks);

	auto& sample = m_sample;
	int n = sample.rows();
	auto dim = sample.cols();

	for (int i = 0; i < n; i++)
	{
		auto partical = m_sample.row(i);

		Progate(partical);
	}

	assert(!m_sample.hasNaN());

	if (m_accelerator == AcceceleratorEnum::GPU)
	{
		Likilihoods_Gpu();

//#if defined(_DEBUG)
//		for (int i = 0; i < n; i++)
//		{
//			auto partical = m_sample.row(i);
//			m_newLiks(i) = Likilihood(i, partical);
//		}
//#endif

	}
#if defined(openMP)
	else if (m_accelerator == AcceceleratorEnum::SMP) {
		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			auto partical = m_sample.row(i);
			m_liks(i) = Likilihood(i, partical);
		}
	}
#endif
	else {
		for (int i = 0; i < n; i++)
		{
			auto partical = m_sample.row(i);
			m_liks(i) = Likilihood(i, partical);
		}
	}

	assert(!m_liks.hasNaN());

	return ExtractMLE();
}

double ParticaleFilterBase::ExtractMLE()
{
	auto& sample = m_sample;
	int n = sample.rows();
	auto dim = sample.cols();

	//m_liks = sample.col(0);
	auto w = m_liks.sum();
	if (!std::isfinite(w))
		w = .0;

	if (w > 0.0001)
	{
		//! Averaging state variable may not be a good choice
		m_waState = (sample.array() * (m_liks / w).cast<ScalarType>().replicate(1, dim).array()).colwise().sum();

		Eigen::Index idx;
		m_liks.maxCoeff(&idx);
		m_mleState = sample.row(idx);
	}
	else // critical bug here, but we will use the mean particle as a dummy
	{
		m_waState = sample.colwise().mean();
		m_mleState = m_waState;
	}

	return w;
}

// resample the weighted sample in O(n*log(n)) time
// generate n ordered point in range [0,1] is n log(n), thus we cannot get any better
void ParticaleFilterBase::Resample(_Out_ LikihoodsType& cdf, _Out_ MatrixType& resampled, _In_ const LikihoodsType& sampleLik, _In_ const MatrixType& sample)
{
	assert((resampled.data() != sample.data()) && "resampled and sample cannot be the same");

	auto n = sample.rows();
	auto dim = sample.cols() - 1;
	resampled.resizeLike(sample);
	cdf.resizeLike(sampleLik);

	std::partial_sum(sampleLik.data(), sampleLik.data() + n, cdf.data());
	cdf /= cdf(n - 1);

	for (int i = 0; i < n; i++)
	{
		// get x from range [0,1] randomly
		auto x = g_uniform(g_rand_mt);

		auto itr = std::lower_bound(cdf.data(), cdf.data() + n, x);
		auto idx = itr - cdf.data();

		resampled.row(i) = sample.row(idx);
	}

	//cdf.array() = 1 / (ScalarType)n;
}

thread_local ArmatureFrame CharacterActionTracker::s_frameCache0;
thread_local ArmatureFrame CharacterActionTracker::s_frameCache1;
