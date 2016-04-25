#include "pch.h"
#include "ClipMetric.h"
#include "CCA.h"
#include "ArmaturePartFeatures.h"

//#include <unsupported\Eigen\fft>
#include "EigenExtension.h"

#define FFTW
#ifdef FFTW
#include <fftw3.h>
#pragma comment(lib, "libfftw3f-3.lib")
#endif

#include <algorithm>
#include <ppl.h>

#include "Causality\Settings.h"

#ifdef _DEBUG
#define DEBUGOUT(x) std::cout << #x << " = " << x << std::endl
#else
#define DEBUGOUT(x)
#endif

using namespace Causality;
using namespace std;
using namespace Concurrency;
using namespace Eigen;
using namespace ArmaturePartFeatures;
using namespace BoneFeatures;

const static double g_Gravity = 9.8;
const static double g_automicLoopCloseDistanceMax = 0.2;

typedef
Weighted<
	RelativeDeformation <
	AllJoints <
	LclRotLnQuatFeature> > >
	CharacterJRSFeature;

typedef
WithVelocity<
	//NormalizeVelocity<
	//Localize<
	EndEffector <
	GblPosFeature >>//>
	PVSFeatureVel;

typedef
//Localize <
EndEffector <
	GblPosFeature >//>
	PVSFeature;


typedef PVSFeature PartsFeatureType;

template <class DerivedX>
void CaculateTimeDirivtiive(MatrixXf& dX, const DenseBase<DerivedX>&X, float frametime, bool isClose = false , int step = 1)
{
	int N = X.rows();
	dX.resizeLike(X);
	if (N > 2)
	{
		dX.middleRows(step, dX.rows() - 2 * step) = (X.topRows(dX.rows() - 2 * step) - X.bottomRows(dX.rows() - 2 * step)) / (2 * (double)step * frametime);
		if (!isClose) // Open Loop case
		{
			for (int d = 1; d < step; d++)
			{
				int i = d;
				dX.row(i) = (X.row(i - d) - X.row(i + d)) / ((double)(2 * d) * frametime);
				i = N - d - 1;
				dX.row(i) = (X.row(i - d) - X.row(i + d)) / ((double)(2 * d) * frametime);
			}
			dX.row(0) = (X.row(0) - X.row(1)) / frametime;
			dX.row(N - 1) = (X.row(N - 2) - X.row(N - 1)) / frametime;
		}
		else // Close Loop case
		{
			dX.topRows(step) = (X.bottomRows(step) - X.middleRows(step, step)) / (2 * (double)step * frametime);;
			dX.bottomRows(step) = (X.middleRows(N - step * 2, step) - X.topRows(step)) / (2 * (double)step * frametime);;
		}
	}
	else
		dX.setZero();
};

void CyclicStreamClipinfo::EnableCyclicMotionDetection(bool is_enable, float cyclicSupportThrehold, float staticEnergyThreshold) {
	m_enableCyclicDtc = is_enable;
	m_staticEnergyThr = staticEnergyThreshold;
	m_cyclicDtcThr = cyclicSupportThrehold;
}

void CyclicStreamClipinfo::EnableCyclicMotionDetection(bool is_enable)
{
	m_enableCyclicDtc = is_enable;
}

void CyclicStreamClipinfo::InitializePvFacade(ShrinkedArmature& parts)
{
	std::lock_guard<std::mutex> guard(m_facadeMutex);
	ClipFacade::SetFeature(m_pFeature);
	ClipFacade::SetActiveEnergy(g_PlayerActiveEnergy, g_PlayerSubactiveEnergy);
	ClipFacade::Prepare(parts, CLIP_FRAME_COUNT * 2, ComputePcaQr | ComputeNormalize | ComputePairDif | ComputeEnergy | LocalizePcaQr);
	ClipFacade::SetEnergyTerms(Ek_TimeDiveritive | Ep_AbsGravity);
	ClipFacade::SetGravityReference(parts.Armature().bind_frame());
	ClipFacade::SetEnergyFilterFunction([this](Eigen::RowVectorXf& Eb) {
		auto& parts = *this->m_pParts;
		Eigen::RowVectorXf Reb(Eb);
		for (int i = 0; i < parts.size(); i++)
		{
			//if (parts[i]->parent() != nullptr)
			//	Eb[i] *= parts[i]->ChainLength / parts[i]->LengthToRoot;
			
			if (parts[i]->parent())
			{
				int pid = parts[i]->parent()->Index;
				Reb[i] = max(Eb[i] - Eb[pid],.0f);
			}
		}
		Eb = Reb;

		if (this->m_confidenceFilter)
			Eb.array() *= this->m_partsConfidences.transpose().array();

		Eb[0] = .0f; // Prevent root to map with an specific part
	});
	
	this->m_partsConfidences.setOnes(parts.size());
}

CharacterClipinfo::CharacterClipinfo()
{
	m_isReady = false;
	m_pParts = nullptr;
}

void CharacterClipinfo::Initialize(const ShrinkedArmature& parts)
{
	m_pParts = &parts;

	auto pRcF = std::make_shared<CharacterJRSFeature>();
	pRcF->SetDefaultFrame(parts.Armature().bind_frame());
	pRcF->InitializeWeights(parts);

	RcFacade.SetFeature(pRcF);

	if (g_UseVelocity)
	{
		auto pPvF = std::make_shared<PVSFeatureVel>();
		//pPvF->SetVelocityThreshold(g_VelocityNormalizeThreshold);
		PvFacade.SetFeature(pPvF);
	}
	else
	{
		auto pPvF = std::make_shared<PVSFeature>();
		PvFacade.SetFeature(pPvF);
	}

	RcFacade.Prepare(parts, -1, ClipFacade::ComputePca | ClipFacade::ComputeEnergy);

	ClipFacade::EnergyFilterFunctionType binded = std::bind(&CharacterClipinfo::FilterLocalRotationEnergy, *this, std::placeholders::_1);
	RcFacade.SetEnergyFilterFunction(std::move(binded));
	PvFacade.Prepare(parts, -1, ClipFacade::ComputePcaQr | ClipFacade::ComputeNormalize | ClipFacade::ComputePairDif | ClipFacade::LocalizePcaQr);
}

void CharacterClipinfo::AnalyzeSequence(array_view<ArmatureFrame> frames, double sequenceTime, bool cyclic)
{
	RcFacade.AnalyzeSequence(frames, sequenceTime, cyclic);
	// Force both facade to use same energy
	PvFacade.SetAllPartsEnengy(RcFacade.GetAllPartsEnergy());
	PvFacade.AnalyzeSequence(frames, sequenceTime, cyclic);
}

CharacterClipinfo::CharacterClipinfo(const ShrinkedArmature& parts)
{
	Initialize(parts);
}

void CharacterClipinfo::SetClipName(const ::std::string& name)
{
	m_clipName = name;
	PvFacade.SetClipName(name);
	RcFacade.SetClipName(name);
}

void CharacterClipinfo::FilterLocalRotationEnergy(Eigen::RowVectorXf & Eb)
{
	auto& parts = *m_pParts;
	for (int i = 0; i < parts.size(); i++)
	{
		// Normalize the angular movenmentum to linear
		Eb[i] *= parts[i]->ChainLength;
	}
}

CyclicStreamClipinfo::~CyclicStreamClipinfo()
{
	if (m_fftplan != nullptr)
	{
		fftwf_free(m_fftplan);
		m_fftplan = nullptr;
	}
}

CyclicStreamClipinfo::CyclicStreamClipinfo(ShrinkedArmature& parts, time_seconds minT, time_seconds maxT, double sampleRateHz, size_t interval_frames)
	: CyclicStreamClipinfo()
{
	Initialize(parts, minT, maxT, sampleRateHz, interval_frames);
}

CyclicStreamClipinfo::CyclicStreamClipinfo()
{
	m_enableCyclicDtc = false;
	m_pParts = nullptr;
	m_fftplan = nullptr;
	m_fillWindowWithFirstFrame = true;
	m_confidenceFilter = true;
	m_minFr = m_maxFr = m_FrWidth = 0;
	m_windowSize = 0;
	m_automaticCloseloop = g_AutomaticLoopClosing;
}

void CyclicStreamClipinfo::Initialize(ShrinkedArmature& parts, time_seconds minT, time_seconds maxT, double sampleRateHz, size_t interval_frames)
{
	InitializeStreamView(parts, minT, maxT, sampleRateHz, interval_frames);

	InitializePvFacade(parts);
}

void CyclicStreamClipinfo::InitializeStreamView(ShrinkedArmature& parts, time_seconds minT, time_seconds maxT, double sampleRateHz, size_t interval_frames)
{
	m_bufferInit = false;
	std::lock_guard<std::mutex> guard(m_bfMutex);

	m_pParts = &parts;
	m_pFeature = make_shared<PartsFeatureType>();

	m_sampleRate = sampleRateHz;

	// find closest 2^k window size, ceil work more robust, but usually it result in 512 frames, which is too long
	m_windowSize = 1 << static_cast<int>(ceil(log2(maxT.count() * sampleRateHz * 5)));

	m_minFr = m_windowSize / (maxT.count() * sampleRateHz);
	assert(m_minFr > 3 && "Miniumal frequency less than 3, the frequency analyze may be in accurate");
	m_minFr = std::max(m_minFr, 3); // Prevent Min Fr be less than 3 that gerneate in invaliad result
	m_maxFr = m_windowSize / (minT.count() * sampleRateHz);
	m_FrWidth = m_maxFr - m_minFr + 1;

	m_analyzeInterval = interval_frames;
	// if automatic, we set 1/16 of period as analyze interval
	if (m_analyzeInterval == 0)
		m_analyzeInterval = std::max(m_windowSize / 32, 1);

	m_frameWidth = 0;
	for (int i = 0; i < parts.size(); i++)
		m_frameWidth += m_pFeature->GetDimension(*parts[i]);

	// make sure each row/column in buffer is aligned as __mm128 for SIMD
	const int alignBoundry = alignof(__m128) / sizeof(float);
	m_bufferWidth = ((m_frameWidth - 1) / alignBoundry + 1) * alignBoundry;
	assert(m_bufferWidth >= m_frameWidth);

	// this 4 is just some majical constant that large enough to avaiod frequent data moving
	m_bufferCapacity = m_windowSize * 4;

	m_buffer.setZero(m_bufferWidth, m_bufferCapacity);
	m_confidencBuffer.setZero(parts.size(), m_bufferCapacity);
	m_Spectrum.setZero(m_bufferWidth, m_windowSize);
	m_SmoothedBuffer.setZero(m_windowSize, m_frameWidth);

	m_cropMargin = 5; // m_sampleRate * 0.333; // 0.1s of frames as margin

	m_cyclicDtcThr = g_RevampActiveSupportThreshold;
	m_staticEnergyThr = g_RevampStaticEnergyThreshold;
	m_whiteNoiseEnergy = g_PlayerTrackingWhiteNoiseEnergy;
	m_pendingFrames = m_sampleRate * g_RevampPendingTime;

	int n = m_windowSize;

#ifdef FFTW
	fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &n, m_frameWidth,
		m_buffer.data(), nullptr, m_bufferWidth, 1,
		(fftwf_complex*)m_Spectrum.data(), nullptr, m_bufferWidth, 1,
		0);
	if (m_fftplan != nullptr)
	{
		fftwf_free(m_fftplan);
		m_fftplan = nullptr;
	}
	m_fftplan = plan;
#endif

	m_bufferHead = m_bufferSize = m_frameCounter = 0;
	m_bufferInit = true;
}

float GetPartConfidence(const ArmaturePart& part, const CyclicStreamClipinfo::FrameType& frame)
{
	//float confi = 0.0f;
	float confi = 1.0f;
	for (auto j : part.Joints)
	{
		int eid = j->ID;

		confi *= frame[eid].GetConfidence();
	}
	
	//confi = powf(confi, 1.0f / part.Joints.size());
	//confi /= part.Joints.size();
	return confi;
}


CyclicStreamClipinfo::RecentFrameResolveResult CyclicStreamClipinfo::StreamFrame(const FrameType & frame)
{
	using namespace Eigen;
	using namespace DirectX;

	RecentFrameResolveResult false_result;
	false_result.MetricReady = false;

	if (!m_bufferInit)
		return false_result;

	assert(
		XMVector3InBounds(
			frame[m_pParts->Armature().root()->ID].GblTranslation,
			XMVectorSplatEpsilon())
		&& "frame must be localized to root");

	bool fillWindow = m_fillWindowWithFirstFrame && m_bufferHead == 0 && m_bufferSize == 0 && m_frameCounter == 0;

	if (m_bufferSize < m_windowSize)
		++m_bufferSize;
	else if (m_bufferSize >= m_windowSize)
	{
		++m_bufferHead;

		if (m_bufferHead + m_bufferSize >= m_bufferCapacity)
		{
			// unique_lock
			std::lock_guard<std::mutex> guard(m_bfMutex);
			// move the buffer to leftmost
			m_buffer.leftCols(m_bufferSize) = m_buffer.middleCols(m_bufferHead, m_bufferSize);
			m_confidencBuffer.leftCols(m_bufferSize) = m_confidencBuffer.middleCols(m_bufferHead, m_bufferSize);
			m_bufferHead = 0;
		}
	}

	// the last column
	int fidx = m_bufferHead + m_bufferSize - 1;
	auto fv = m_buffer.col(fidx);

	auto& parts = *m_pParts;
	int stIdx = 0;

	
	auto identilizedFrame = IdentilizeFrame(parts, frame);

	for (int i = 0; i < parts.size(); i++)
	{
		auto& part = *parts[i];

		m_confidencBuffer(i, fidx) = GetPartConfidence(part, frame);
		auto bv = m_pFeature->Get(part, identilizedFrame);
		int dim = m_pFeature->GetDimension(*parts[i]);
		fv.segment(stIdx, dim) = bv.transpose();
		stIdx += dim;
	}

	if (fillWindow && m_bufferHead == 0 && m_bufferSize == 1 && m_frameCounter == 0)
	{
		m_buffer.middleCols(m_bufferHead + 1, m_windowSize - 1) = fv.replicate(1, m_windowSize - 1);
		m_confidencBuffer.middleCols(m_bufferHead + 1, m_windowSize - 1) = m_confidencBuffer.col(fidx).replicate(1, m_windowSize - 1);
		m_bufferSize = m_windowSize;
	}

	++m_frameCounter;
	if (m_enableCyclicDtc && 
		(m_frameCounter > m_pendingFrames) &&
		(m_frameCounter % m_analyzeInterval == 0) && 
		m_bufferSize >= m_windowSize)
	{
		return AnaylzeRecentAction();
	}
	
	false_result.ConfidenceReady = false;
	false_result.BufferingProgress = ((float)m_frameCounter / m_pendingFrames);
	false_result.BufferingReady = m_frameCounter >= m_pendingFrames;
	return false_result;
}

CyclicStreamClipinfo::FrameType CyclicStreamClipinfo::IdentilizeFrame(const ShrinkedArmature & parts, const ArmatureFrame & frame)
{
	FrameType result = frame;
	auto& armature = parts.Armature();
	auto& rotBone = frame[armature.root()->ID];
	XMVECTOR rootTran = rotBone.GblRotation;
	for (int i = 0; i < armature.size(); i++)
	{
		result[i].GblTranslation -= rootTran;
	}
	return result;
}

void CyclicStreamClipinfo::ResetStream()
{
	std::lock_guard<std::mutex> guard(m_bfMutex);
	m_bufferHead = m_bufferSize = m_frameCounter = 0;
}

std::mutex & CyclicStreamClipinfo::AqucireFacadeMutex()
{
	return m_facadeMutex;
}

CyclicStreamClipinfo::RecentFrameResolveResult CyclicStreamClipinfo::AnaylzeRecentAction(RecentAcrtionBehavier forceBehavier)
{
	// Anaylze starting
	int head = m_bufferHead + m_bufferSize - m_windowSize - 1;
	int tail = m_bufferHead + m_bufferSize;

	int windowSize = m_windowSize;
	int swSize = std::min(windowSize / 2, m_pendingFrames);

	m_isStaticPose = false;
	CaculateSpecturum(head, windowSize);

	RecentFrameResolveResult result;
	auto fr = CaculatePeekFrequency(m_Spectrum);
	result.SetFrequencyResolveResult(fr);
	m_highFreqNoiseEnergy = fr.NoiseEnergy;

	// Using latest buffered data
	result.AnotherEnergy = CaulateKinectEnergy(tail - swSize, swSize, Ek_MotionRange);
	float nenerg = (fr.Energy - m_whiteNoiseEnergy) / (m_staticEnergyThr);

	// Prevent low energy high-frequency perodic motion
	result.PeriodicConfidence = (fr.Support / m_cyclicDtcThr) * std::min(1.0f, nenerg * 5.0f);
	result.StaticConfidence = (1.0f - nenerg);
	result.Behavier = 
		forceBehavier ? forceBehavier :
		result.PeriodicConfidence >= 1.0f ?
		RecentActionBehavier_PeriodMotion : 
		result.StaticConfidence >= 1.0f ?
		RecentActionBehavier_FreezedPose :
		RecentActionBehavier_Auto;
	result.BufferingProgress = 1.0f;
	result.MetricReady = false;
	result.ConfidenceReady = true;
	result.BufferingReady = true;

	if (result.Behavier != RecentActionBehavier_Auto)
	{
		if (m_facadeMutex.try_lock()) {
			lock_guard<mutex> guard(m_facadeMutex, std::adopt_lock);

			if (result.Behavier == RecentActionBehavier_PeriodMotion)
			{
				auto& X = ClipFacade::SetFeatureMatrix();
				float Tseconds = 1 / fr.Frequency;

				m_energyTerms &= ~(Ep_AbsGravity | Ep_SampleMeanLength);
				m_energyTerms |= Ek_TimeDiveritive;

				m_isClose = false;
				int period = fr.PeriodInFrame;
				CropResampleInput(X, tail - period, period, CLIP_FRAME_COUNT, g_InputSmoothStrength);
				ClipFacade::SetClipTime(Tseconds);
			}
			else
			{
				m_energyTerms &= ~(Ek_SampleVarience | Ek_TimeDiveritive);
				m_energyTerms |= Ep_AbsGravity;

				m_isClose = false;
				ClipFacade::SetFeatureMatrix(GetLatestPoseFeatureFrame());
				ClipFacade::SetClipTime(.0);
			}

			ClipFacade::CaculatePartsMetric();

			result.MetricReady = true;
		}
		else
		{
			cout << "Assignment process failed due to facade busy" << fr.Support << endl;
		}
	}

	return result;
}

bool CyclicStreamClipinfo::AnaylzeRecentPose()
{
	m_isStaticPose = true;
	if (m_facadeMutex.try_lock()) {
		lock_guard<mutex> guard(m_facadeMutex, std::adopt_lock);

		ClipFacade::SetFeatureMatrix(GetLatestPoseFeatureFrame());

		ClipFacade::SetClipTime(.0);
		ClipFacade::CaculatePartsMetric();
	}
	return true;
}

void CyclicStreamClipinfo::CaculateSpecturum(size_t head, size_t windowSize)
{
	m_readerHead = head;
	m_readerSize = windowSize;

	std::lock_guard<std::mutex> guard(m_bfMutex);

	int n = windowSize;
#ifdef FFTW
	fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &n, m_frameWidth,
		m_buffer.col(head).data(), nullptr, m_bufferWidth, 1,
		(fftwf_complex*)m_Spectrum.data(), nullptr, m_bufferWidth, 1,
		0);

	// keep the plan alive will help the plan speed
	if (m_fftplan)
		fftwf_destroy_plan((fftwf_plan)m_fftplan);

	m_fftplan = plan;

	fftwf_execute(plan);
#endif
}

void CyclicStreamClipinfo::CropResampleInput(_Out_ MatrixXf& X, size_t head, size_t inputPeriod, size_t resampledPeriod, float smoothStrength)
{
	const int smoothIteration = g_InputSmoothIteration;
	auto T = inputPeriod;

	int margin = std::min((int)head, (int)m_cropMargin);
	assert(margin > 0);

	int marginedHead = head - margin;
	int inputLength = T + margin;

	//assert(inputLength < m_bufferSize);
	// copy the buffer
	auto Xs = m_SmoothedBuffer.topRows(inputLength);


	{
		// Critial section, copy data from buffer and transpose in Column Major
		std::lock_guard<std::mutex> guard(m_bfMutex);
		Xs = m_buffer.block(0, marginedHead, m_frameWidth, inputLength).transpose();
		m_partsConfidences = m_confidencBuffer.middleCols(marginedHead, inputLength).rowwise().mean();
	}

	if (m_automaticCloseloop && T > 5)
	{
		// assumes a linear noise from e0 to e1, moves e1 to e0 and distribute the changes with in the loop
		auto e0 = Xs.topRows(margin).colwise().mean();
		auto e1 = Xs.bottomRows(margin).colwise().mean();
		auto tune = (e0 - e1).eval();

		float endDivergence = tune.norm();
		if (endDivergence > g_automicLoopCloseDistanceMax * sqrt(ceilf((float)m_frameWidth / 3.0f)))
		{
			std::cout << "[Warning] Failed to cloose loop : Loop Divergence = " << endDivergence << endl;
			//m_isClose = false;
		}

		Xs.middleRows(margin, T) += VectorXf::LinSpaced(T, .0f, 1.0f).asDiagonal() * tune.replicate(T, 1);
		//Xs.bottomRows(m_cropMargin) = Xs.topRows(m_cropMargin);
		m_isClose = true;
	}
	else
	{
		m_isClose = false;
	}

	// Smooth the input
	// * Since we copied the margin, no need to hint 'CloseLoop' in the smooth
	laplacianSmooth(Xs, smoothStrength, smoothIteration, Eigen::OpenLoop);

	//! To-do , use better method to crop out the "example" single period

	if (X.rows() != resampledPeriod)
		X.resize(resampledPeriod, m_frameWidth);

	// Resample input into X
	cublicBezierResample(X,
		m_SmoothedBuffer.middleRows(margin, T),
		resampledPeriod,
		Eigen::CloseLoop);
}

CyclicStreamClipinfo::FrequencyResolveResult CyclicStreamClipinfo::CaculatePeekFrequency(const Eigen::MatrixXcf & spectrum)
{
	FrequencyResolveResult fr;

	// Column major spectrum, 1 column = 1 frame in time
	auto& Xf = spectrum;
	auto windowSize = Xf.cols();

	int idx;
	auto& Ea = m_SpectrumEnergy;

	// Note Xf is (bufferWidth X windowSize)
	// thus we crop it top frameWidth rows and intersted band in cols to caculate energy
	int extfrCut = min((int)(m_maxFr * 2), (int)Xf.cols());
	auto Eall = Xf.block(0, 0, m_frameWidth, extfrCut).cwiseAbs2().colwise().maxCoeff().eval();
	Eall /= (m_sampleRate * m_windowSize);

	if (g_PeriodAnalyzeAggreateWithMax)
		Ea = Xf.block(0, m_minFr - 1, m_frameWidth, m_FrWidth + 2)
		.cwiseAbs2().colwise().maxCoeff().transpose();
	else
		Ea = Xf.block(0, m_minFr - 1, m_frameWidth, m_FrWidth + 2)
		.cwiseAbs2().colwise().sum().transpose() / m_frameWidth;
	// Normalize the spectrum energy, as the fftw does not applies the term dt/N
	Ea /= (m_sampleRate * m_windowSize);

	Ea(0) = .0f;
	Ea(Ea.size() - 1) = .0f;

	DEBUGOUT(Eall);

	Ea.segment(1, Ea.size() - 2).maxCoeff(&idx); // Frequency 3 - 30
	++idx;

	// get the 2 adjicant freequency as well, to perform interpolation to get better estimation
	auto Ex = Ea.segment<3>(idx - 1).eval();
	idx += m_minFr;

	DEBUGOUT(Ex.transpose());

	Vector3f Ix = { idx - 1.0f, (float)idx, idx + 1.0f };
	float peekFreq = Ex.dot(Ix) / Ex.sum();
	
	int T = (int)ceil(windowSize / peekFreq);

	float snr = Ex.sum() / Ea.sum();
	assert(snr <= 1.0f);

	fr.Frequency = windowSize / peekFreq / m_sampleRate;
	fr.PeriodInFrame = T;
	fr.Support = snr /** snr*/;
	fr.Energy = Eall.segment(m_minFr, m_FrWidth).sum();
	fr.NoiseEnergy = Eall.segment(m_minFr + m_FrWidth, Eall.cols() - (m_minFr + m_FrWidth)).sum();
	return fr;
}

float CyclicStreamClipinfo::CaulateKinectEnergy(size_t head, size_t windowSize, EnergyTermEnum term)
{
	float Ek = .0f;
	if (windowSize > 1)
	{
		std::lock_guard<std::mutex> guard(m_bfMutex);
		m_partsConfidences = m_confidencBuffer.middleCols(head, windowSize).rowwise().mean();
		auto raw = m_buffer.block(0, head, m_frameWidth, windowSize);
		float frametime = 1 / m_sampleRate;

		if (term & Ek_SampleVarience)
		{
			auto mean = raw.rowwise().mean();
			Ek = (raw - mean.replicate(1, windowSize)).squaredNorm()
				/(m_frameWidth * (windowSize - 1));
		}
		else if (term & Ek_TimeDiveritive)
		{
			CaculateTimeDirivtiive(m_bufferDx, raw, frametime, false, m_step);
			laplacianSmooth(m_bufferDx, 0.8, 2, m_isClose ? CloseLoop : OpenLoop);
			Ek = m_bufferDx.squaredNorm() / (m_frameWidth * windowSize);
		}
		else if (term & Ek_MotionRange)
		{
			VectorXf rang = (raw.rowwise().maxCoeff() - raw.rowwise().minCoeff());
			Ek = reshape(rang, 3, -1).colwise().norm().dot(m_partsConfidences);
		}
	}
	return Ek;
}

ClipFacade::ClipFacade()
{
	m_pParts = nullptr;
	m_flag = NotInitialize;
	m_energyTerms = Ek_SampleVarience;
	m_pairInfoLevl = NonePair;
	m_dimP = -1;
	m_pdFix = false;
	m_inited = false;
	m_isClose = false;
	m_step = 1;
	m_pcaCutoff = g_CharacterPcaCutoff;
	m_ActiveEnergyThreshold = g_CharacterActiveEnergy;
	m_SubactiveEnergyThreshold = g_CharacterSubactiveEnergy;
}

ClipFacade::~ClipFacade()
{

}

void ClipFacade::SetActiveEnergy(float active, float subActive) {
	m_ActiveEnergyThreshold = active; m_SubactiveEnergyThreshold = subActive;
}

void ClipFacade::SetGravityReference(ArmatureFrameConstView frame)
{
	auto& parts = *m_pParts;
	int stIdx = 0;
	m_refX.resize(m_partSt.back() + m_partDim.back());
	for (int i = 0; i < parts.size(); i++)
	{
		auto& part = *parts[i];
		auto bv = m_pFeature->Get(part, frame);
		int dim = m_pFeature->GetDimension(*parts[i]);
		m_refX.segment(stIdx, dim) = bv.transpose();
		stIdx += dim;
	}

	// Automatic gravity mask
	if (GetAllPartDimension() == 3 && m_energyTerms & Ep_AbsGravity)
	{
		m_Gmask = RowVector3f(0.0, g_Gravity, 0.0).replicate(1, m_pParts->size());
	}
}

void ClipFacade::SetGravityMask(const Eigen::RowVectorXf gmask)
{
	m_Gmask = gmask;
}

void ClipFacade::Prepare(const ShrinkedArmature & parts, int clipLength, int flag)
{
	assert(m_pFeature != nullptr && "Set Feature Before Call Prepare");

	m_pParts = &parts;
	m_flag = flag;

	m_Edim.resize(parts.size());
	m_Eb.setOnes(parts.size());
	m_partDim.resize(parts.size());
	m_partSt.resize(parts.size());

	m_ActiveParts.reserve(parts.size());
	m_SubactiveParts.reserve(parts.size());

	if (m_flag & ComputePca)
	{
		m_Pcas.resize(parts.size());
		m_PcaDims.setConstant(parts.size(), -1);

		if (m_flag & ComputePcaQr)
			m_thickQrs.resize(parts.size());
	}

	m_partSt[0] = 0;
	m_pdFix = true;

	for (int i = 0; i < parts.size(); i++)
	{
		m_partDim[i] = m_pFeature->GetDimension(*parts[i]);
		if (i > 0)
		{
			m_pdFix = m_pdFix && (m_partDim[i] == m_partDim[i - 1]);
			m_partSt[i] = m_partSt[i - 1] + m_partDim[i - 1];
		}

		m_Edim[i].setOnes(m_partDim[i]);
	}
	m_dimP = m_pdFix ? m_partDim[0] : -1;

	int fLength = m_partDim.back() + m_partSt.back();

	auto& dframe = parts.Armature().bind_frame();
	SetGravityReference(dframe);

	if (clipLength > 0)
	{
		m_X.resize(clipLength, fLength);
		m_uX.resize(fLength);
		m_cX.resizeLike(m_X);
		m_dX.resizeLike(m_X);
		if (m_flag & ComputeNormalize)
			m_Xnor.resizeLike(m_X);
	}

	if (!m_pdFix)
		m_flag &= ~ComputePairDif;

	if (m_flag & ComputePairDif)
	{
		m_difMean.setZero(parts.size() * m_dimP, parts.size());
		m_difCov.setZero(parts.size() * m_dimP, parts.size() * m_dimP);
	}
}

void ClipFacade::SetComputationFlags(int flags)
{
	m_flag = flags;
}

void ClipFacade::SetEnergyTerms(unsigned energyTerms) { m_energyTerms = energyTerms; }

void ClipFacade::AnalyzeSequence(array_view<const ArmatureFrame> frames, double sequenceTime, bool cyclic)
{
	assert(m_pParts != nullptr);

	m_clipTime = sequenceTime;

	m_isClose = cyclic;
	//? HACK! IS BAD.
	SetFeatureMatrix(frames, sequenceTime, cyclic);

	CaculatePartsMetric();
}

void ClipFacade::SetFeatureMatrix(array_view<const ArmatureFrame> frames, double duration, bool cyclic)
{
	assert(m_pParts != nullptr && m_flag != NotInitialize);

	m_inited = false;

	auto& parts = *m_pParts;
	int fLength = m_partDim.back() + m_partSt.back();

	auto dt = duration / (frames.size() - (size_t)(!cyclic));
	bool useVelocity = dt > 0;

	m_X.resize(frames.size(), fLength);
	for (int f = 0; f < frames.size(); f++)
	{
		auto& lastFrame = frames[f > 0 ? f - 1 : frames.size() - 1];
		auto& frame = frames[f];
		for (int i = 0; i < parts.size(); i++)
		{
			auto part = parts[i];

			auto fv = m_X.block(f, m_partSt[i], 1, m_partDim[i]);

			if (!useVelocity)
				fv = m_pFeature->Get(*part, frame);
			else
				fv = m_pFeature->Get(*part, frame, lastFrame, dt);
		}
	}
}

void ClipFacade::CaculatePartsMetric()
{
	auto& parts = *m_pParts;

	m_uX = m_X.colwise().mean().eval();
	m_cX = m_X - m_uX.replicate(m_X.rows(), 1).eval();
	if (m_flag & ComputeNormalize)
		m_Xnor = m_X;

	m_Edim.resize(parts.size());
	for (int i = 0; i < parts.size(); i++)
		m_Edim[i].setOnes(m_partDim[i]);

	if (m_flag & ComputeEnergy)
		m_Eb.setZero(parts.size());

	m_partDim.resize(parts.size());
	m_partSt.resize(parts.size());
	m_Pcas.resize(parts.size());
	m_thickQrs.resize(parts.size());

	// Caculate Numberical difference of X 
	double frametime = m_clipTime / (double)m_X.rows();
	CaculateTimeDirivtiive(m_dX, m_X, frametime, m_isClose, m_step);

	if (m_dX.rows() > 2)
		laplacianSmooth(m_dX, 0.8, 2, m_isClose ? CloseLoop : OpenLoop);

	//int N = m_X.rows();
	//m_dX.resizeLike(m_X);
	//if (N > 2)
	//{
	//	m_dX.middleRows(m_step, m_dX.rows() - 2 * m_step) = (m_X.topRows(m_dX.rows() - 2 * m_step) - m_X.bottomRows(m_dX.rows() - 2 * m_step)) / (2 * (double)m_step * frametime);
	//	if (!m_isClose) // Open Loop case
	//	{
	//		for (int d = 1; d < m_step; d++)
	//		{
	//			int i = d;
	//			m_dX.row(i) = (m_X.row(i - d) - m_X.row(i + d)) / ((double)(2 * d) * frametime);
	//			i = N - d - 1;
	//			m_dX.row(i) = (m_X.row(i - d) - m_X.row(i + d)) / ((double)(2 * d) * frametime);
	//		}
	//		m_dX.row(0) = (m_X.row(0) - m_X.row(1)) / frametime;
	//		m_dX.row(N - 1) = (m_X.row(N - 2) - m_X.row(N - 1)) / frametime;
	//	}
	//	else // Close Loop case
	//	{
	//		m_dX.topRows(m_step) = (m_X.bottomRows(m_step) - m_X.middleRows(m_step, m_step)) / (2 * (double)m_step * frametime);;
	//		m_dX.bottomRows(m_step) = (m_X.middleRows(N - m_step * 2, m_step) - m_X.topRows(m_step)) / (2 * (double)m_step * frametime);;
	//	}
	//}
	//else
	//	m_dX.setZero();

	if (m_flag & ComputeNormalize)
		for (int i = 0; i < parts.size(); i++)
		{
			m_Xnor.middleCols(m_partSt[i], m_partDim[i]).rowwise().normalize();
		}

	if (m_flag & ComputeEnergy)
	{
		for (int i = 0; i < parts.size(); i++)
		{

			if (m_energyTerms & Ek_SampleVarience)
			{
				m_Edim[i] = m_cX.middleCols(m_partSt[i], m_partDim[i]).cwiseAbs2().colwise().sum().transpose() / m_X.rows();
				m_Eb[i] = m_Edim[i].sum();
			}
			else if (m_energyTerms & Ek_TimeDiveritive)
			{
				auto cols = m_dX.middleCols(m_partSt[i], m_partDim[i]);
				m_Edim[i] = cols.cwiseAbs2().colwise().mean().transpose();
				m_Eb[i] = m_Edim[i].sum();
			}

			if (m_energyTerms & Ep_SampleMeanLength)
				m_Eb[i] += m_uX.segment(m_partSt[i], m_partDim[i]).cwiseAbs2().sum();
			else if (m_energyTerms & Ep_AbsGravity)
				m_Eb[i] += ((m_uX.segment(m_partSt[i], m_partDim[i])
					- m_refX.segment(m_partSt[i], m_partDim[i])).array()
					* m_Gmask.segment(m_partSt[i], m_partDim[i]).array())
				.cwiseAbs().sum();

			m_Edim[i] = m_Edim[i].array().sqrt();
			m_Eb[i] = sqrtf(m_Eb[i]);
		}

		if (m_energyFilter != nullptr)
		{
			m_energyFilter(m_Eb);

			for (int i = 0; i < parts.size(); i++)
			{
				m_Edim[i].setConstant(m_Eb[i]);
			}
		}
	}

	assert(m_Eb.size() == parts.size() && "Energy function must be compute or imported before further processing");

	float maxEnergy = m_Eb.maxCoeff();

	m_ActiveParts.clear();
	m_SubactiveParts.clear();
	m_PcaDims.setConstant(-1);

	for (int i = 0; i < parts.size(); i++)
	{
		// Prevent root part become an active part
		if (i > 0)
		{
			if (m_Eb[i] >= m_ActiveEnergyThreshold * maxEnergy)
			{
				m_ActiveParts.push_back(i);
			}
			else if (m_Eb[i] >= m_SubactiveEnergyThreshold * maxEnergy)
			{
				m_SubactiveParts.push_back(i);
			}
		}
	}

	if (m_flag & ComputePairDif)
		CaculatePartsPairMetric();

	// Compute Pca for all active and sub-active parts
	//x inactive parts
	for (int i = 0; i < parts.size(); i++)
	{
		if ((m_flag & ComputePca) && (m_Eb[i] >= m_SubactiveEnergyThreshold * maxEnergy))
		{
			CaculatePartPcaQr(i);
		}
	}

	m_inited = true;
}

void ClipFacade::CaculatePartPcaQr(int i)
{
	auto& pca = m_Pcas[i];

	auto& parts = *m_pParts;
	if (!(LocalizePcaQr & m_flag) || !parts[i]->parent())
	{
		pca.computeCentered(m_cX.middleCols(m_partSt[i], m_partDim[i]), true);
		pca.setMean(m_uX.segment(m_partSt[i], m_partDim[i]));
	}
	else // Force the pca caculated is based on the relative sequence
	{
		pca.compute(GetPartsDifferenceSequence(i, parts[i]->parent()->Index));
	}

	auto d = pca.reducedRank(m_pcaCutoff);
	m_PcaDims[i] = d;

	//! Potiential unnessary matrix copy here!!!
	if (m_flag & ComputePcaQr)
		m_thickQrs[i].compute(m_Pcas[i].coordinates(d), false, true);
}

void ClipFacade::CaculatePartsPairMetric(PairDifLevelEnum level)
{
	m_pairInfoLevl = level;

	auto& parts = *m_pParts;

	m_difMean.resize(parts.size() * m_dimP, parts.size());
	m_difCov.resize(parts.size() * m_dimP, parts.size() * m_dimP);

	float thrh = 0;
	switch (level)
	{
	case ClipFacade::ActivePartPairs:
		thrh = m_ActiveEnergyThreshold;
		break;
	case ClipFacade::SubactivePartPairs:
		thrh = m_SubactiveEnergyThreshold;
		break;
	case ClipFacade::AllPartPairs:
		thrh = 0;
		break;
	case ClipFacade::NonePair:
	default:
		return;
	}

	thrh *= m_Eb.maxCoeff();

	MatrixXf Xij(ClipFrames(), m_dimP);
	RowVectorXf uXij(m_dimP);

	for (int i = 0; i < parts.size(); i++)
	{
		if (m_Eb[i] < thrh) continue;

		auto pi = parts[i]->parent();
		if (pi)
			CaculatePairMetric(Xij, i, pi->Index, uXij);

		for (int j = i + 1; j < parts.size(); j++)
		{
			// Always caculate the metric of a part to its parent
			if ((m_Eb[j] < thrh))
				continue;

			CaculatePairMetric(Xij, i, j, uXij);
		}
	}
}

inline void ClipFacade::CaculatePairMetric(Eigen::MatrixXf &Xij, int i, int j, Eigen::RowVectorXf &uXij)
{
	Xij = GetPartsDifferenceSequence(i, j);
	uXij = Xij.colwise().mean();
	m_difMean.block(i*m_dimP, j, m_dimP, 1) = uXij.transpose();

	auto covij = m_difCov.block(i*m_dimP, j*m_dimP, m_dimP, m_dimP);

	Xij -= uXij.replicate(Xij.rows(), 1);
	covij.noalias() = Xij.transpose() * Xij;

	// mean is aniti-symetric, covarience is symetric
	m_difMean.block(j*m_dimP, i, m_dimP, 1) = -uXij.transpose();
	m_difCov.block(j*m_dimP, i*m_dimP, m_dimP, m_dimP) = covij;
}

CyclicStreamClipinfo::RecentFrameResolveResult::RecentFrameResolveResult()
{
	ZeroMemory(this, sizeof(RecentFrameResolveResult));
	ConfidenceReady = false;
	MetricReady = false;
	BufferingReady = false;
}

void CyclicStreamClipinfo::RecentFrameResolveResult::SetFrequencyResolveResult(const FrequencyResolveResult & fr) {
	static_cast<FrequencyResolveResult&>(*this) = fr;
}
