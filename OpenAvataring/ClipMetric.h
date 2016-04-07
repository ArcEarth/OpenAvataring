#pragma once
#include <atomic>
#include <mutex>
#include <functional>

#include <Eigen\Dense>

#include "ArmatureParts.h"
#include "CCA.h"

#include "Causality\Animations.h"

namespace std
{
	class thread;
}

namespace Causality
{
	class CcaArmatureTransform;

	// A ClipFacade is a anayls base on certian given feature
	class ClipFacade
	{
	public:
		enum ComputationFlag
		{
			NotInitialize = 0,
			ComputeBasic = 1,
			ComputeNormalize = 0x2,
			ComputePca = 0x4,
			ComputePcaQr = 0x8 | ComputePca,
			ComputePairDif = 0x10,
			ComputeVelocity = 0x20,
			ComputeEnergy = 0x40,
			ComputeAll = ComputePcaQr | ComputeNormalize | ComputePairDif | ComputeEnergy,
		};

		enum PairDifLevelEnum
		{
			NonePair = 0,
			ActivePartPairs = 1,
			SubactivePartPairs = 2,
			AllPartPairs = 3,
		};

		enum EnergyTermEnum
		{
			Ek_SampleVarience = 1,
			Ep_SampleMeanLength = 2,
			Ek_TimeDiveritive = 4, //This enery are direved from X(t)-X(t-1), time dependent
			Ep_AbsGravity = 8, // This term compute the hight (Y) energy from the default pose
			Ek_MotionRange = 16, // max(X) - min(X)
		};

		typedef std::function<void(Eigen::RowVectorXf&)> EnergyFilterFunctionType;
		// Construction & Meta-data acess
	public:
		ClipFacade();
		~ClipFacade();

		const ShrinkedArmature&			ArmatureParts() const { return *m_pParts; }

		auto&							ActiveParts() const { return m_ActiveParts; }
		auto&							SubactiveParts() const { return m_SubactiveParts; }
		void							SetActiveEnergy(float active, float subActive);

		const auto&						ClipName() const { return m_clipName; }
		void							SetClipName(const std::string& name) { m_clipName = name; }
		int								ClipFrames() const { return m_X.rows(); }
		// Clip time length in seconds
		double							ClipTime() const { return m_clipTime; }
		void							SetClipTime(double time_seconds) { m_clipTime = time_seconds; }

		bool							IsReady() const { return m_inited; }
		// return -1 if variable feature dimension
		int								GetAllPartDimension() const { return m_pdFix ? m_dimP : -1; }
		PairDifLevelEnum				GetPartPairInfoLevel() const { return m_pairInfoLevl; }

		float							PcaCutoff() const { return m_pcaCutoff; }
		void							SetPcaCutoff(float cutoff) { m_pcaCutoff = cutoff; }

		shared_ptr<IArmaturePartFeature>GetFeature() const { return m_pFeature; }
		void							SetFeature(const std::shared_ptr<IArmaturePartFeature> & pFeature)
		{
			m_pFeature = std::move(pFeature);
		}

		template <typename _Ty, class... _Types>
		void							SetFeature(_Types&&... _Args)
		{
			m_pFeature = std::make_shared<_Ty, _Types...>( _STD forward<_Types>(_Args)... );
		}

		void							SetGravityReference(ArmatureFrameConstView frames);
		void							SetGravityMask(const Eigen::RowVectorXf gmask);

		void							Prepare(const ShrinkedArmature& parts, int clipLength = -1, int flag = ComputeAll);
		void							SetComputationFlags(int flags);

		void							SetEnergyTerms(unsigned energyTerms);
		void							SetEnergyFilterFunction(const EnergyFilterFunctionType &filter)
		{
			m_energyFilter = filter;
		}
		void							SetEnergyFilterFunction(EnergyFilterFunctionType &&filter)
		{
			m_energyFilter = std::move(filter);
		}

		void							AnalyzeSequence(array_view<const ArmatureFrame> frames, double sequenceTime, bool cyclic);

		void							SetFeatureMatrix(array_view<const ArmatureFrame> frames, double duration, bool cyclic);
		void							SetFeatureMatrix(const Eigen::MatrixXf& X) { m_inited = false; m_X = X; }
		void							SetFeatureMatrix(Eigen::MatrixXf&& X) { m_inited = false; m_X = std::move(X); }
		Eigen::MatrixXf&				SetFeatureMatrix() { m_inited = false; return m_X; }

		// Caculate energy and Pca and Qr for parts
		void							CaculatePartsMetric();

		void							CaculatePartPcaQr(int pid);

	protected:
		void							CaculatePartsPairMetric(PairDifLevelEnum level = ActivePartPairs);

		void							CaculatePairMetric(Eigen::MatrixXf &Xij, int i, int j, Eigen::RowVectorXf &uXij);

		// Part-wise metric accessers
	public:
		Eigen::DenseIndex				GetPartStartIndex(int pid) const
		{
			return m_pdFix ? pid * m_dimP : m_partSt[pid];
		}

		Eigen::DenseIndex				GetPartDimension(int pid) const
		{
			return m_pdFix ? m_dimP : m_partDim[pid];
		}

		float							GetPartEnergy(int pid) const { return m_Eb[pid]; }

		void							SetPartEnergy(int pid, float value) { m_Eb[pid] = value; }

		auto&							GetAllPartsEnergy() const
		{
			return m_Eb;
		}

		template <class Deirved>
		inline void						SetAllPartsEnengy(const Eigen::DenseBase<Deirved>& eb)
		{
			m_Eb = eb;
		}

		auto&							GetPartDimEnergy(int pid) const
		{
			return m_Edim[pid];
		}

		const Eigen::MatrixXf&			GetAllPartsSequence() const
		{
			return m_X;
		}
		auto							GetPartSequence(int pid) const
		{
			return m_X.middleCols(GetPartStartIndex(pid), GetPartDimension(pid));
		}

		auto&							GetAllPartsNormalizedSequence() const
		{
			return m_Xnor;
		}
		auto							GetPartNormalizedSequence(int pid) const
		{
			return m_Xnor.middleCols(GetPartStartIndex(pid), GetPartDimension(pid));
		}

		auto							GetPartMean(int pid) const
		{
			return m_uX.segment(GetPartStartIndex(pid), GetPartDimension(pid));
		}
		auto&							GetAllPartsMean() const
		{
			return m_uX;
		}

		// Only Active and Subactive part have Pca / Qr
		int								GetPartPcaDim(int pid) const { return m_PcaDims[pid]; }
		auto&							GetPartPca(int pid) const
		{
			return m_Pcas[pid];
		}
		auto							GetPartPcadSequence(int pid, int d = -1) const
		{
			if (d < 0)
				d = m_PcaDims[pid];
			return m_Pcas[pid].coordinates(d);
		}
		// QrView ofpPart Pcad Sequence
		auto							GetPartPcadQrView(int pid, int stFrame = 0, int frames = -1) const
		{
			return Eigen::QrView<Eigen::MatrixXf>(m_thickQrs[pid], stFrame, frames);
		}

		auto							GetPartsDifferenceSequence(int pi, int pj) const
		{
			//assert(m_partDim[pi] == m_partDim[pj]);
			return GetPartSequence(pi) - GetPartSequence(pj);
		}

		// return the row vector of E(X(pi)-X(pj))
		auto							GetPartsDifferenceMean(int pi, int pj) const
		{
			return m_difMean.block(pi*m_dimP, pj, m_dimP, 1).transpose();
		}
		auto							GetPartsDifferenceCovarience(int pi, int pj) const
		{
			return m_difCov.block(pi*m_dimP, pj*m_dimP, m_dimP, m_dimP);
		}

	protected:
		std::string				m_clipName;

		const ShrinkedArmature*	m_pParts;

		std::shared_ptr<IArmaturePartFeature>
								m_pFeature;

		bool					m_pdFix;		// Is Part feature fixed size
		std::vector<int>		m_partSt;		// Part feature start index for part i
		std::vector<int>		m_partDim;		// Part feature dimension for part i
		int						m_dimP;			// Part feature dimension

		double					m_clipTime;
		unsigned				m_flag;
		unsigned				m_energyTerms;
		PairDifLevelEnum		m_pairInfoLevl;
		bool					m_isClose;		// flag if the clip is cyclic (or close)
		bool					m_inited;
		int						m_step;		// Frame difference for caculating velocity

		// Feature data matrix
		Eigen::MatrixXf			m_X;	// Raw data
		Eigen::RowVectorXf		m_uX;	// X mean
		Eigen::MatrixXf			m_cX;	// Centered X
		Eigen::MatrixXf			m_dX;	// Centre differcence of X

		Eigen::MatrixXf			m_Xnor;	// Partiwise Rowwise noramlized X

		Eigen::RowVectorXf		m_refX;
		Eigen::RowVectorXf		m_Gmask;
		Eigen::RowVectorXf		m_Eb;	// 1xB, Blockwise Energy
		std::vector<Eigen::VectorXf>	
								m_Edim;// Dimension energy for parts i , for variable size

		Eigen::VectorXi			m_PcaDims;		
		std::vector<Eigen::Pca<Eigen::MatrixXf>>
								m_Pcas;			// Pca of the raw data, partiwise

		std::vector<Eigen::QrStore<Eigen::MatrixXf>>
								m_thickQrs;		// Qr of the Pca of the raw data, partiwise

		float					m_pcaCutoff;

		std::vector<int>		m_ActiveParts;  // it's a set
		std::vector<int>		m_SubactiveParts;

		float					m_ActiveEnergyThreshold;
		float					m_SubactiveEnergyThreshold;

		Eigen::MatrixXf			m_difMean;
		Eigen::MatrixXf			m_difCov;

		EnergyFilterFunctionType m_energyFilter;
	};

	class CharacterClipinfo
	{
	public:
		ClipFacade RcFacade;
		ClipFacade PvFacade;

		CharacterClipinfo();

		void Initialize(const ShrinkedArmature& parts);

		void AnalyzeSequence(array_view<ArmatureFrame> frames, double sequenceTime, bool cyclic);

		explicit CharacterClipinfo(const ShrinkedArmature& parts);

		void							SetArmatureParts(const ShrinkedArmature& parts) { m_pParts = &parts; }
		const ShrinkedArmature&			ArmatureParts() const { return *m_pParts; }
		auto&							ActiveParts() const { return	PvFacade.ActiveParts(); }
		auto&							SubactiveParts() const { return PvFacade.SubactiveParts(); }

		const std::string&				ClipName() const { return m_clipName; }

		void SetClipName(const ::std::string& name);

		int								ClipFrames() const { return PvFacade.ClipFrames(); }
		// Clip time length in seconds
		double							ClipTime() const { return PvFacade.ClipTime(); }

		bool							IsReady() const { return RcFacade.IsReady() && PvFacade.IsReady(); }

		void							FilterLocalRotationEnergy(Eigen::RowVectorXf& Eb);
	protected:
		std::string				m_clipName;
		const ShrinkedArmature*	m_pParts;
		bool					m_isReady;
	};

	enum RecentAcrtionBehavier
	{
		RecentActionBehavier_Auto = 0,
		RecentActionBehavier_FreezedPose = 1,
		RecentActionBehavier_PeriodMotion = 2,
	};

	class CyclicStreamClipinfo : protected ClipFacade
	{
	public:
		struct FrequencyResolveResult
		{
			float Frequency;
			float Support;
			float Energy;
			float NoiseEnergy; // Energy in high frequency domin
			int	  PeriodInFrame;
		};

		struct RecentFrameResolveResult : public FrequencyResolveResult
		{
			RecentFrameResolveResult();

			void SetFrequencyResolveResult(const FrequencyResolveResult& fr);

			RecentAcrtionBehavier Behavier;
			float AnotherEnergy;
			float PeriodicConfidence;
			float StaticConfidence;
			float BufferingProgress;
			bool  MetricReady;
			bool  ConfidenceReady;
			bool  BufferingReady;
		};

		typedef IArmatureStreamAnimation::frame_type FrameType;
		~CyclicStreamClipinfo();
		// set interval_frames == 0 to automaticly estimate based on windows size and sample rate
		CyclicStreamClipinfo(ShrinkedArmature& parts, time_seconds minT, time_seconds maxT, double sampleRateHz, size_t interval_frames = 0);
		CyclicStreamClipinfo();

		void Initialize(ShrinkedArmature& parts, time_seconds minT, time_seconds maxT, double sampleRateHz, size_t interval_frames = 0);

		// Important, input frame use this method
		// return true if a new metric of ClipFacade is avaiable
		RecentFrameResolveResult StreamFrame(const FrameType& frame);

		FrameType IdentilizeFrame(const Causality::ShrinkedArmature & parts, const Causality::ArmatureFrame & frame);

		void ResetStream();

		std::mutex& AqucireFacadeMutex();

		ClipFacade& AsFacade()
		{
			return *this;
		}

		const ClipFacade& AsFacade() const
		{
			return *this;
		}

		void EnableCyclicMotionDetection(bool is_enable, float cyclicSupportThrehold, float staticEnergyThreshold);
		void EnableCyclicMotionDetection(bool is_enable = true);

		RecentFrameResolveResult AnaylzeRecentAction(RecentAcrtionBehavier forceBehavier = RecentActionBehavier_Auto);
		bool AnaylzeRecentPose();

		auto GetLatestPoseFeatureFrame() {
			return m_buffer.col(m_bufferHead + m_bufferSize - 1).transpose();
		}

	protected:
		void InitializePvFacade(ShrinkedArmature& parts);
		void InitializeStreamView(ShrinkedArmature& parts, time_seconds minT, time_seconds maxT, double sampleRateHz, size_t interval_frames);

		// Result stored in this->m_Spectrum
		void CaculateSpecturum(size_t head, size_t windowSize);

		// Crop and resample the interst segment in the input stream
		// Returns 2TxD Matrix stores input features partwise, T == framePerCycle
		// Result is stored in this->X
		void CropResampleInput(_Out_ Eigen::MatrixXf& X, size_t head, size_t inputPeriod, size_t framePerCycle, float smoothStrength);

		FrequencyResolveResult CaculatePeekFrequency(const Eigen::MatrixXcf& spectrum);
		float CaulateKinectEnergy(size_t head, size_t windowSize, EnergyTermEnum term = Ek_SampleVarience);
	private:
		double		m_minT, m_maxT;
		double		m_sampleRate;
		int			m_cropMargin;
		bool		m_isStaticPose;
		bool		m_fillWindowWithFirstFrame;
		bool		m_automaticCloseloop;
		bool		m_confidenceFilter;

		// min Frequency, in Frames unit
		int			m_minFr, m_maxFr, m_FrWidth;
		int			m_frameWidth; // logical feature frame width, in floats
		int			m_windowSize; // expected windows size

		int			m_analyzeInterval; // analyze should invoke for every $m_analyzeInterval$ frames arrived

		std::shared_ptr<IArmaturePartFeature>
					m_pFeature;
		int			m_featureDim;

		std::atomic_bool m_bufferInit;
		std::atomic_bool m_enableCyclicDtc;
		int			m_frameCounter;
		float		m_cyclicDtcThr; // The threshold to classify as Cyclic motion
		float		m_staticEnergyThr;
		float		m_whiteNoiseEnergy;
		float		m_highFreqNoiseEnergy;
		int			m_pendingFrames;

		// thread sychronization
		//boost::icl::interval_set<int> m_bufferAccess;
		std::mutex	m_bfMutex;	// buffer access mutex
		int			m_readerHead;
		int			m_readerSize;

		// Feature buffer
		//! Column majored feature matrix, 1 column = 1 frame in time
		Eigen::MatrixXf		m_buffer;
		//! Column majored confidence matrix, 1 column = 1 frame in time
		//! sizeof(Parts) X sizeof(Frames)
		Eigen::MatrixXf		m_confidencBuffer;

		std::mutex			m_spMutex;
		//! Column major spectrum, 1 column = 1 frame in time
		Eigen::MatrixXcf	m_Spectrum;	// wSize x fSize
		Eigen::MatrixXf		m_bufferDx;
		Eigen::VectorXf		m_SpectrumEnergy;//In each frequency
		Eigen::VectorXf		m_EnergyStatisticFilter; // An Low-Pass Gaussian Filter
		Eigen::MatrixXf		m_SmoothedBuffer;
		Eigen::VectorXf		m_partsConfidences; // Average Tracking confidences of the corresponding part in current analyzed sequence

		int					m_bufferHead;
		int					m_bufferSize;
		int					m_bufferWidth;	// actual floats per frame in buffer, considered with alignments
		int					m_bufferCapacity;

		void*				m_fftplan;		// impleamentation detail, handle to fft

		std::mutex			m_facadeMutex;	// Facade access mutex
	};
}
