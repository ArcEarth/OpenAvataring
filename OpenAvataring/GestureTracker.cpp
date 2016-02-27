#include "pch.h"
#include "GestureTracker.h"
#include <numeric>
#include <random>
#ifndef _DEBUG
#define openMP
#endif
using namespace Causality;

std::random_device g_rand;
std::mt19937 g_rand_mt(g_rand());
static std::normal_distribution<IGestureTracker::ScalarType> g_normal_dist(0, 1);
static std::uniform_real<IGestureTracker::ScalarType> g_uniform(0, 1);

IGestureTracker::~IGestureTracker()
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

ParticaleFilterBase::ScalarType ParticaleFilterBase::StepParticals()
{
	Resample(m_newLiks, m_newSample, m_liks, m_sample);
	m_newSample.swap(m_sample);
	m_newLiks.swap(m_liks);

	auto& sample = m_sample;
	int n = sample.rows();
	auto dim = sample.cols();

#ifdef _AMP_GPU_PARA_
	concurrency::array_view<ScalarType, 2> sampleView(concurrency::extent<2>(n,dim + 1), m_sample.data());
	concurrency::array_view<float, 2> animationView(concurrency::extent<2>());
	concurrency::parallel_for_each(concurrency::extent<1>(n),
		[sampleView](concurrency::index<1> idx) restrict(amp)
	{
		
	});
#else
#if defined(openMP)
#pragma omp parallel for
#endif
	for (int i = 0; i < n; i++)
	{
		auto partical = m_sample.row(i);

		Progate(partical);
		m_liks(i) = Likilihood(i, partical);
		//m_sample(i, 0) *= Likilihood(i, partical);
	}
#endif

	return ExtractMLE();
}

double ParticaleFilterBase::ExtractMLE()
{
	auto& sample = m_sample;
	int n = sample.rows();
	auto dim = sample.cols();

	//m_liks = sample.col(0);
	auto w = m_liks.sum();
	if (w > 0.0001)
	{
		//! Averaging state variable may not be a good choice
		m_waState = (sample.array() * (m_liks / w).cast<ScalarType>().replicate(1, dim).array()).colwise().sum();

		Eigen::Index idx;
		m_liks.maxCoeff(&idx);
		m_mleState = sample.row(idx);
	}
	else // critial bug here, but we will use the mean particle as a dummy
	{
		m_waState = sample.colwise().mean();
		m_mleState = m_waState;
	}

	return w;
}

// resample the weighted sample in O(n*log(n)) time
// generate n ordered point in range [0,1] is n log(n), thus we cannot get any better
void ParticaleFilterBase::Resample(_Out_ LikihoodsType& resampledLik, _Out_ MatrixType& resampled, _In_ const LikihoodsType& sampleLik, _In_ const MatrixType& sample)
{
	assert((resampled.data() != sample.data()) && "resampled and sample cannot be the same");

	auto n = sample.rows();
	auto dim = sample.cols() - 1;
	resampled.resizeLike(sample);
	resampledLik.resizeLike(sampleLik);

	auto& cdf = resampledLik;
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

	cdf.array() = 1 / (ScalarType)n;
}