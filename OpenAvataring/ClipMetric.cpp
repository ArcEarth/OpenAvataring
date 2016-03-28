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

void CyclicStreamClipinfo::EnableCyclicMotionDetection(bool is_enable, float cyclicSupportThrehold) {
	m_enableCyclicDtc = is_enable;
	m_cyclicDtcThr = cyclicSupportThrehold;
}

void CyclicStreamClipinfo::InitializePvFacade(ShrinkedArmature& parts)
{
	std::lock_guard<std::mutex> guard(m_facadeMutex);
	ClipFacade::SetFeature(m_pFeature);
	ClipFacade::SetActiveEnergy(g_PlayerActiveEnergy, g_PlayerSubactiveEnergy);
	ClipFacade::Prepare(parts, CLIP_FRAME_COUNT * 2, ComputePcaQr | ComputeNormalize | ComputePairDif | ComputeEnergy);
	ClipFacade::SetEnergyTerms(Ek_TimeDiveritive /*| Ep_AbsGravity*/);
	ClipFacade::SetGravityReference(parts.Armature().bind_frame());
	ClipFacade::SetEnergyFilterFunction([&parts](Eigen::RowVectorXf& Eb) {
		for (int i = 0; i < parts.size(); i++)
		{
			if (parts[i]->parent() != nullptr)
				Eb[i] *= parts[i]->ChainLength / parts[i]->LengthToRoot;
		}
	});
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
	PvFacade.Prepare(parts, -1, ClipFacade::ComputePcaQr | ClipFacade::ComputeNormalize | ClipFacade::ComputePairDif);
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
{
	m_enableCyclicDtc = false;
	m_pParts = nullptr;
	m_fftplan = nullptr;
	m_minFr = m_maxFr = m_FrWidth = 0;
	m_windowSize = 0;
	Initialize(parts, minT, maxT, sampleRateHz, interval_frames);
}

CyclicStreamClipinfo::CyclicStreamClipinfo()
{
	m_enableCyclicDtc = false;
	m_pParts = nullptr;
	m_fftplan = nullptr;
	m_minFr = m_maxFr = m_FrWidth = 0;
	m_windowSize = 0;
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
	m_windowSize = 1 << static_cast<int>(floor(log2(maxT.count() * sampleRateHz * 4)));

	m_minFr = m_windowSize / (maxT.count() * sampleRateHz);
	m_maxFr = m_windowSize / (minT.count() * sampleRateHz);
	m_FrWidth = m_maxFr - m_minFr + 1;

	m_analyzeInterval = interval_frames;
	// if automatic, we set 1/16 of period as analyze interval
	if (m_analyzeInterval == 0)
		m_analyzeInterval = std::max(m_windowSize / 16, 1);

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
	m_Spectrum.setZero(m_bufferWidth, m_windowSize);
	m_SmoothedBuffer.setZero(m_windowSize, m_frameWidth);

	m_cropMargin = m_sampleRate * 0.5; // 0.1s of frames as margin
	m_cyclicDtcThr = g_RevampActiveSupportThreshold;

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

bool CyclicStreamClipinfo::StreamFrame(const FrameType & frame)
{
	using namespace Eigen;
	using namespace DirectX;
	if (!m_bufferInit)
		return false;

	assert(
		XMVector3InBounds(
			frame[m_pParts->Armature().root()->ID].GblTranslation,
			XMVectorSplatEpsilon())
		&& "frame must be localized to root");

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
			m_bufferHead = 0;
		}
	}

	// the last column
	auto fv = m_buffer.col(m_bufferHead + m_bufferSize - 1);

	auto& parts = *m_pParts;
	int stIdx = 0;
	for (int i = 0; i < parts.size(); i++)
	{
		auto& part = *parts[i];
		auto bv = m_pFeature->Get(part, frame);
		int dim = m_pFeature->GetDimension(*parts[i]);
		fv.segment(stIdx, dim) = bv.transpose();
		stIdx += dim;
	}

	++m_frameCounter;
	if (m_enableCyclicDtc && m_frameCounter >= m_analyzeInterval && m_bufferSize >= m_windowSize)
	{
		m_frameCounter = 0;
		return AnaylzeRecentStream();
	}

	return false;
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

bool CyclicStreamClipinfo::AnaylzeRecentStream()
{
	// Anaylze starting
	size_t head = m_bufferHead;
	size_t windowSize = m_windowSize;

	CaculateSpecturum(head, windowSize);

	auto fr = CaculatePeekFrequency(m_Spectrum);

	if (fr.Support > m_cyclicDtcThr)
	{
		if (m_facadeMutex.try_lock()) {
			lock_guard<mutex> guard(m_facadeMutex, std::adopt_lock);

			auto& X = ClipFacade::SetFeatureMatrix();
			float Tseconds = 1 / fr.Frequency;
			CropResampleInput(X, head, fr.PeriodInFrame, CLIP_FRAME_COUNT, 0.8f);

			ClipFacade::SetClipTime(Tseconds);
			ClipFacade::CaculatePartsMetric();

			return true;
		}
		else
		{
			cout << "Assignment process failed due to facade busy" << fr.Support << endl;
		}

	}
	else {
		cout << "Assignment process reject, peak frequency support = " << fr.Support << endl;
	}
	return false;
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
	const int smoothIteration = 4;
	auto T = inputPeriod;

	int inputLength = T + m_cropMargin * 2;
	assert(inputLength < m_bufferSize);
	// copy the buffer
	auto Xs = m_SmoothedBuffer.topRows(inputLength);

	{
		// Critial section, copy data from buffer and transpose in Column Major
		std::lock_guard<std::mutex> guard(m_bfMutex);
		Xs = m_buffer.block(0, head + m_windowSize - inputLength, m_frameWidth, inputLength).transpose();
	}

	// Smooth the input 
	laplacianSmooth(Xs, smoothStrength, smoothIteration, Eigen::OpenLoop);

	//! To-do , use better method to crop out the "example" single period

	if (X.rows() != resampledPeriod)
		X.resize(resampledPeriod, m_frameWidth);
	// Resample input into X
	cublicBezierResample(X,
		m_SmoothedBuffer.middleRows(m_cropMargin, T),
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
	Ea = Xf.block(0, m_minFr - 1, m_frameWidth, m_FrWidth + 2).cwiseAbs2().colwise().sum().transpose();

	DEBUGOUT(Ea.transpose());

	Ea.segment(1, Ea.size() - 2).maxCoeff(&idx); // Frequency 3 - 30
	++idx;

	// get the 2 adjicant freequency as well, to perform interpolation to get better estimation
	auto Ex = Ea.segment<3>(idx - 1);
	idx += m_minFr;

	DEBUGOUT(Ex.transpose());

	Vector3f Ix = { idx - 1.0f, (float)idx, idx + 1.0f };
	float peekFreq = Ex.dot(Ix) / Ex.sum();

	int T = (int)ceil(windowSize / peekFreq);

	float snr = Ex.sum() / Ea.segment(1, Ea.size() - 2).sum();

	fr.Frequency = windowSize / peekFreq / m_sampleRate;
	fr.PeriodInFrame = T;
	fr.Support = snr;

	return fr;
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
		m_Gmask = Vector3f(0.0, g_Gravity, 0.0).replicate(1, m_pParts->size()).transpose();
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
	int N = m_X.rows();
	m_dX.resizeLike(m_X);
	m_dX.middleRows(m_step, m_dX.rows() - 2 * m_step) = (m_X.topRows(m_dX.rows() - 2 * m_step) - m_X.bottomRows(m_dX.rows() - 2 * m_step)) / (2 * (double)m_step * frametime);

	if (!m_isClose) // Open Loop case
	{
		for (int d = 1; d < m_step; d++)
		{
			int i = d;
			m_dX.row(i) = (m_X.row(i - d) - m_X.row(i + d)) / ((double)(2 * d) * frametime);
			i = N - d - 1;
			m_dX.row(i) = (m_X.row(i - d) - m_X.row(i + d)) / ((double)(2 * d) * frametime);
		}
		m_dX.row(0) = (m_X.row(0) - m_X.row(1)) / frametime;
		m_dX.row(N - 1) = (m_X.row(N - 2) - m_X.row(N - 1)) / frametime;
	}
	else // Close Loop case
	{
		m_dX.topRows(m_step) = (m_X.bottomRows(m_step) - m_X.middleRows(m_step, m_step)) / (2 * (double)m_step * frametime);;
		m_dX.bottomRows(m_step) = (m_X.middleRows(N-m_step*2, m_step) - m_X.topRows(m_step)) / (2 * (double)m_step * frametime);;
	}

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
				m_Edim[i] = m_cX.middleCols(m_partSt[i], m_partDim[i]).cwiseAbs2().colwise().sum().transpose();
				m_Eb[i] = m_Edim[i].sum();
			}
			else if (m_energyTerms & Ek_TimeDiveritive)
			{
				auto cols = m_cX.middleCols(m_partSt[i], m_partDim[i]);
				m_Edim[i] = cols.cwiseAbs2().colwise().sum().transpose();
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
			if (m_Eb[i] > m_ActiveEnergyThreshold * maxEnergy)
			{
				m_ActiveParts.push_back(i);
			}
			else if (m_Eb[i] > m_SubactiveEnergyThreshold * maxEnergy)
			{
				m_SubactiveParts.push_back(i);
			}
		}

		// Compute Pca for all active and sub-active parts
		// inactive parts
		if ((m_flag & ComputePca) && (m_Eb[i] > m_SubactiveEnergyThreshold * maxEnergy))
		{
			CaculatePartPcaQr(i);
		}
	}

	if (m_flag & ComputePairDif)
		CaculatePartsPairMetric();

	m_inited = true;
}

void ClipFacade::CaculatePartPcaQr(int i)
{
	auto& pca = m_Pcas[i];

	pca.computeCentered(m_cX.middleCols(m_partSt[i], m_partDim[i]), true);
	pca.setMean(m_uX.segment(m_partSt[i], m_partDim[i]));
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
