#pragma once
#include "GestureTracker.h"

namespace Causality
{
	class PartilizedTransformer;

	class CharacterActionTracker : public ParticaleFilterBase
	{
	public:
		CharacterActionTracker(const ArmatureFrameAnimation& animation, const PartilizedTransformer &transfomer);
		// Inherited via ParticaleFilterBase
	public:
		virtual void	Reset(const InputVectorType & input) override;

		void			Reset();

		ScalarType		Step(const InputVectorType& input, ScalarType dt) override;

		void			GetScaledFrame(_Out_ ArmatureFrameView frame, ScalarType t, ScalarType s) const;

		void			SetLikihoodVarience(const InputVectorType& v);

		void			SetTrackingParameters(ScalarType stdevDVt, ScalarType varVt, ScalarType stdevDs, ScalarType varS);
		void			SetVelocityTolerance(ScalarType thrVt);
		void			SetScaleTolerance(ScalarType thrS);

		void			SetStepSubdivition(int subdiv);

		void			SetParticalesSubdiv(int timeSubdiv, int scaleSubdiv, int vtSubdiv);

		const ArmatureFrameAnimation& Animation() const { return m_Animation; }
	protected:
		void			StepParticalsGPU();
		void			SetInputState(const InputVectorType & input, ScalarType dt) override;
		ScalarType		Likilihood(int idx, const TrackingVectorBlockType & x) override;
		void			Progate(TrackingVectorBlockType & x) override;

		InputVectorType GetCorrespondVector(const TrackingVectorBlockType & x, ArmatureFrameView frameCache0, ArmatureFrameView frameChache1) const;

	protected:
		const ArmatureFrameAnimation&	m_Animation;
		const PartilizedTransformer&	m_Transformer;

		struct GpuImpl;
		uptr<GpuImpl>					m_gpu;
		//mutable vector<ArmatureFrame>	m_Frames;
		//mutable vector<ArmatureFrame>	m_LastFrames;


		InputVectorType					m_CurrentInput;
		std::shared_ptr<IArmaturePartFeature>	m_pFeature;

		bool							m_currentValiad;
		int								m_stepSubdiv;
		int								m_tSubdiv;
		int								m_vtSubdiv;
		int								m_scaleSubdiv;
		// Likilihood distance cov 
		int								m_lidxCount;
		MatrixType						m_fvectors;
		InputVectorType					m_LikCov;
		// time difference
		ScalarType						m_dt;
		ScalarType						m_confidentThre;

		// Progation velocity variance
		ScalarType						m_stdevDVt;
		ScalarType						m_varVt;
		ScalarType						m_stdevDs;
		ScalarType						m_varS;
		ScalarType						m_uS;
		ScalarType						m_thrVt;
		ScalarType						m_thrS;

		static constexpr size_t FrameCacheSize = 100;
		static thread_local Bone s_frameCache0[FrameCacheSize], s_frameCache1[FrameCacheSize];
	};
}