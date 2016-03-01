#pragma once
#include <Eigen\Core>
#include <Causality\Animations.h>
#include <Causality\RenderSystemDecl.h>
#include "ArmatureParts.h"

namespace Causality
{
	class PartilizedTransformer;

	class IGestureTracker
	{
	public:
	public:
		typedef float ScalarType;
		typedef double LikilihoodScalarType;
		typedef Eigen::Matrix<LikilihoodScalarType, -1, 1> LikihoodsType;
		typedef Eigen::Matrix<ScalarType, -1, -1, Eigen::AutoAlign | Eigen::RowMajor> MatrixType;
		typedef Eigen::Matrix<ScalarType, 1, -1, Eigen::AutoAlign | Eigen::RowMajor> TrackingVectorType;
		typedef Eigen::Block<MatrixType, 1, -1, true> TrackingVectorBlockType;

		typedef Eigen::Matrix<ScalarType, 1, -1> InputVectorType;

		virtual ~IGestureTracker();

	public:
		// aka, Initialize, discard all history information, just initialize with input, reset all state
		virtual void Reset(const InputVectorType& input) = 0;
		// Step forward the tracking state from t to t+1
		// return the confident of current tracking
		virtual ScalarType Step(const InputVectorType& input, ScalarType dt) = 0;
		// Get the tracking state
		virtual const TrackingVectorType& CurrentState() const = 0;
	};

	// Provide base methods for Particle Filter
	// 
	// The sample matrix S is N x (Dim+1)
	// Where N is number of Particals
	// Dim is the state vector dimension
	// S.col(0), the first column of S stores the weights
	// S.row(i) is a particale
	// S(i, 0) is the particale weight
	// S(i, 1...Dim) is the state vector
	class ParticaleFilterBase : public IGestureTracker
	{
	public:
		enum AcceceleratorEnum
		{
			None,	// Sequential excution
			SMP,	// CPU Parallelism
			GPU,	// C++ AMP Impl
		};

	public:
		ParticaleFilterBase();
		~ParticaleFilterBase();

		ScalarType Step(const InputVectorType& input, ScalarType dt) override;

		// Weighted Avaerage State
		const TrackingVectorType& CurrentState() const override;
		const TrackingVectorType& MostLikilyState() const;

		int*  GetTopKStates(int k) const;

		//virtual void Reset(const InputVectorType& input) = 0;

		const MatrixType& GetSampleMatrix() const;
		const LikihoodsType& GetSampleLikilihoods() const;

	protected: // Interfaces
		virtual void SetInputState(const InputVectorType& input, ScalarType dt) = 0;
		// Get the likilihood of partical state x in current time with pre-seted input state
		virtual LikilihoodScalarType Likilihood(int idx, const TrackingVectorBlockType &x) = 0;

		virtual void Progate(TrackingVectorBlockType& x) = 0;

		virtual void Likilihoods_Gpu();

	protected:
		ScalarType StepParticals();

		double ExtractMLE();

		void Resample(_Out_ LikihoodsType& resampledLik, _Out_ MatrixType& resampled, _In_ const LikihoodsType& sampleLik, _In_ const MatrixType& sample);

		AcceceleratorEnum	m_accelerator;
		int					m_maxK;
		mutable vector<int> m_srtIdxes;
		LikihoodsType		m_liks;
		LikihoodsType		m_newLiks;
		MatrixType			m_sample;
		MatrixType			m_newSample;
		// Mean state 
		TrackingVectorType	m_waState;
		TrackingVectorType	m_mleState;
	};

	namespace Internal
	{
		struct CharacterActionTrackerGpuImpl;
	}

	class CharacterActionTracker : public ParticaleFilterBase
	{
	public:
		CharacterActionTracker(const ArmatureFrameAnimation& animation, const PartilizedTransformer &transfomer, IRenderDevice* pDevice = nullptr);
		~CharacterActionTracker();
		// This is evil, but nessary to host in an vector
		CharacterActionTracker(const CharacterActionTracker& rhs);
		// Inherited via ParticaleFilterBase
	public:
		void			SwitchAccelerater(AcceceleratorEnum acc);

		virtual void	Reset(const InputVectorType & input) override;

		void			Reset();

		void			SetupGpuImplParameters();

		ScalarType		Step(const InputVectorType& input, ScalarType dt) override;

		void			GetScaledFrame(_Out_ ArmatureFrameView frame, ScalarType t, ScalarType s) const;

		void			SetLikihoodVarience(const InputVectorType& v);

		void			SetTrackingParameters(ScalarType VtProgate, ScalarType varVt, ScalarType ScaleProgate, ScalarType varS);
		void			SetVelocityTolerance(ScalarType thrVt);
		void			SetScaleTolerance(ScalarType thrS);

		void			SetStepSubdivition(int subdiv);

		void			SetParticalesSubdiv(int timeSubdiv, int scaleSubdiv, int vtSubdiv);

		const ArmatureFrameAnimation& Animation() const { return m_Animation; }
	protected:
		void			SetInputState(const InputVectorType & input, ScalarType dt) override;
		LikilihoodScalarType
						Likilihood(int idx, const TrackingVectorBlockType & x) override;
		void			Progate(TrackingVectorBlockType & x) override;
		void			Likilihoods_Gpu() override;

		InputVectorType GetCorrespondVector(const TrackingVectorBlockType & x, ArmatureFrameView frameCache0, ArmatureFrameView frameChache1) const;

	protected:
		const ArmatureFrameAnimation&	m_Animation;
		const PartilizedTransformer&	m_Transformer;
		const ShrinkedArmature&			m_parts;
		std::vector<const ArmaturePart*>m_effectors;

		using GpuImpl = Internal::CharacterActionTrackerGpuImpl;
		uptr<GpuImpl>					m_gpu;
		IRenderDevice*					m_pDevice;
		struct GpuAcceleratorView;
		//static wptr<GpuAcceleratorView>	s_accView;
		//sptr<GpuAcceleratorView>		m_accView;

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
		LikilihoodScalarType			m_confidentThre;

		// Progation velocity variance
		ScalarType						m_VtProgation;
		ScalarType						m_varVt;
		ScalarType						m_ScaleProgation;
		ScalarType						m_varS;
		ScalarType						m_uS;
		ScalarType						m_uVt;
		ScalarType						m_thrVt;
		ScalarType						m_thrS;

		size_t							m_framesCounter;

		static constexpr size_t FrameCacheSize = 100;
		static thread_local ArmatureFrame s_frameCache0, s_frameCache1;
	};
}