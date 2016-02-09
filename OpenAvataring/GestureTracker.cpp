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
		return m_sample(i, 0) > m_sample(j, 0);
	});
	return m_srtIdxes.data();
}

const ParticaleFilterBase::MatrixType & ParticaleFilterBase::GetSampleMatrix() const { return m_sample; }

ParticaleFilterBase::ScalarType ParticaleFilterBase::StepParticals()
{
	Resample(m_newSample, m_sample);
	m_newSample.swap(m_sample);

	auto& sample = m_sample;
	int n = sample.rows();
	auto dim = sample.cols() - 1;

#if defined(openMP)
#pragma omp parallel for
#endif
	for (int i = 0; i < n; i++)
	{
		auto partical = m_sample.block<1, -1>(i, 1, 1, dim);

		Progate(partical);
		m_sample(i, 0) *= Likilihood(i, partical);
	}

	m_liks = sample.col(0);
	auto w = sample.col(0).sum();
	if (w > 0.0001)
	{
		//! Averaging state variable may not be a good choice
		m_waState = (sample.rightCols(dim).array() * sample.col(0).replicate(1, dim).array()).colwise().sum();
		m_waState /= w;
		
		Eigen::Index idx;
		sample.col(0).maxCoeff(&idx);
		m_mleState = sample.block<1, -1>(idx, 1,1,sample.cols()-1);
	}
	else // critial bug here, but we will use the mean particle as a dummy
	{
		m_waState = sample.rightCols(dim).colwise().mean();
		m_mleState = m_waState;
	}

	return w;
}

// resample the weighted sample in O(n*log(n)) time
// generate n ordered point in range [0,1] is n log(n), thus we cannot get any better
void ParticaleFilterBase::Resample(MatrixType & resampled, const MatrixType & sample)
{
	assert((resampled.data() != sample.data()) && "resampled and sample cannot be the same");

	auto n = sample.rows();
	auto dim = sample.cols() - 1;
	resampled.resizeLike(sample);

	auto cdf = resampled.col(0);
	std::partial_sum(sample.col(0).data(), sample.col(0).data() + n, cdf.data());
	cdf /= cdf[n - 1];

	for (int i = 0; i < n; i++)
	{
		// get x from range [0,1] randomly
		auto x = g_uniform(g_rand_mt);

		auto itr = std::lower_bound(cdf.data(), cdf.data() + n, x);
		auto idx = itr - cdf.data();

		resampled.block<1, -1>(i, 1, 1, dim) = sample.block<1, -1>(idx, 1, 1, dim);
	}

	cdf.array() = 1 / (ScalarType)n;
}