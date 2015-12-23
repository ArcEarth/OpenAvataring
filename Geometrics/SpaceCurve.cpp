#define NOMINMAX
#include "SpaceCurve.h"

using namespace Geometrics;
using namespace DirectX;


SpaceCurve::SpaceCurve() {
	m_isClose = false;
}

SpaceCurve::~SpaceCurve() {}

SpaceCurve::SpaceCurve(const gsl::span<Vector3>& trajectory, bool closeLoop/* = false*/)
{
	m_isClose = closeLoop;
	for (auto& point : trajectory)
	{
		this->push_back(point);
	}
}


std::vector<Vector3> SpaceCurve::FixCountSampling(unsigned int SampleSegmentCount, bool Smooth/* = true*/) const
{
	if (size() <= 1) {
		return std::vector<Vector3>();
	}
	float Interval = length() / SampleSegmentCount;
	// Allocate C+1 size array for storage the result
	std::vector<Vector3> sample(SampleSegmentCount + 1);
	Vector3 ptr;
	float r = 0.0f;
	unsigned int k = 1;

	for (unsigned int i = 0; i < SampleSegmentCount + 1; i++, r += Interval)
	{
		sample[i] = position(r);
	}

	if (Smooth)
		laplacianSmooth<Vector3>(sample, 0.8f, 4, m_isClose);

	return sample;
}

std::vector<Vector3> SpaceCurve::FixCountSampling2(unsigned int SampleSegmentCount) const
{
	auto trajectory = FixCountSampling(SampleSegmentCount * 15, true);
	SpaceCurve smoothSampler(trajectory);
	return smoothSampler.FixCountSampling(SampleSegmentCount, false);
}

std::vector<Vector3> Geometrics::SpaceCurve::sample(size_t sampleCount) const
{
	return FixCountSampling2(sampleCount);
}

void SpaceCurve::resample(size_t anchorCount, bool smooth)
{
	auto sample = FixCountSampling2(anchorCount);

	m_anchors.resize(sample.size());
	for (int i = 0; i < sample.size(); i++)
		reinterpret_cast<Vector3&>(m_anchors[i]) = sample[i];

	if (m_isClose)
		m_anchors.push_back(m_anchors[0]);

	updateLength();
}

void Geometrics::SpaceCurve::smooth(float alpha, unsigned iteration)
{
	laplacianSmooth<Vector4>( gsl::span<Vector4>((Vector4*)data(), size()), alpha, iteration, m_isClose);
	updateLength();
}

std::vector<Vector3> SpaceCurve::FixIntervalSampling(float Interval) const
{
	assert(Interval != 0.0f);
	float r = length();
	unsigned int Count = (int)(r / Interval) + 1;
	auto trajectory = FixCountSampling(Count * 15, true);
	SpaceCurve smoothSampler(trajectory);
	r = smoothSampler.length();
	Count = (int)(r / Interval) + 1;
	return smoothSampler.FixCountSampling(Count, false);
}

void SpaceCurve::setClose(bool close) {
	if (!m_isClose && close && !empty())
	{
		auto btr = XMLoadA(m_anchors.back());
		m_anchors.push_back(m_anchors[0]);
		auto vtr = XMLoadA(m_anchors[0]);
		btr = btr - vtr;
		btr += XMVector3Length(btr);
		m_anchors.back().w = XMVectorGetW(btr);
	}

	m_isClose = close; 
}

size_t SpaceCurve::size() const { 
	return m_anchors.empty() ? 0 : m_anchors.size() - m_isClose; }

float SpaceCurve::length() const
{
	if (m_anchors.empty()) return 0.0f;
	return m_anchors.back().w;
}

bool XM_CALLCONV SpaceCurve::push_back(FXMVECTOR vtr, bool force)
{
	float len = .0f;
	if (m_anchors.empty()) {
		m_anchors.emplace_back();
		XMStoreA(m_anchors.back(), vtr);
		m_anchors.back().w = .0f;
	}
	else
	{
		XMVECTOR btr = XMLoadA(m_anchors[size() - 1]);
		len = XMVectorGetW(btr + XMVector3Length(vtr - btr));

		// ingnore duplicated anchors
		if (!force && abs(len - m_anchors.back().w) < XM_EPSILON * 8)
			return false;

		if (!m_isClose)
			m_anchors.emplace_back();

		XMStoreA(m_anchors.back(), vtr);
		m_anchors.back().w = len;
	}

	if (m_isClose)
	{
		XMVECTOR v = XMLoadA(m_anchors[0]);
		len = len + XMVectorGetW(XMVector3Length(v - vtr));
		XMStoreA(m_anchors.back(), v);
		m_anchors.back().w = len;
	}
	return true;
}

XMVECTOR Geometrics::SpaceCurve::back() const { return XMLoadFloat4A(&m_anchors[size() - 1]); }

bool SpaceCurve::push_back(const Vector3& p,bool force)
{
	return push_back(XMLoad(p),force);
}

// Tangent vector at anchor index
XMVECTOR SpaceCurve::extract(float t) const
{
	float l = length();
	if (m_isClose)
	{
		t = fmodf(t, l);
		if (t < .0f)
			t += l;
	}
	else
	{
		// clamp to [0,l]
		t = std::max(.0f, std::min(t, l));
	}

	unsigned int a = 0, b = m_anchors.size() - 1;
	while (b - a > 1)
	{
		unsigned int k = (a + b) / 2;
		if (m_anchors[k].w > t) b = k;
		else a = k;
	}

	XMVECTOR v0 = XMLoadFloat4A(&m_anchors[a]);
	XMVECTOR v1 = XMLoadFloat4A(&m_anchors[b]);
	float rt = (t - m_anchors[a].w) / (m_anchors[b].w - m_anchors[a].w);
	XMVECTOR v2 = XMVectorLerp(v0, v1, rt);

	// set w = 1.0
	v2 = XMVectorSelect(v2, g_XMIdentityR3.v, g_XMIdentityR3.v);
	return v2;
}

XMVECTOR SpaceCurve::tangent(int idx) const
{
	using namespace DirectX;
	int pid = idx - (idx > 0);
	int rid = idx + (idx + 1 < m_anchors.size());

	XMVECTOR v0 = XMLoadFloat4A(&m_anchors[pid]);
	XMVECTOR v1 = XMLoadFloat4A(&m_anchors[rid]);
	v1 = XMVectorSubtract(v1, v0);
	v0 = XMVectorSplatW(v1);
	if (XMVector4Less(v0, g_XMEpsilon.v))
	{
		v1 = XMVectorZero();
	}
	else
	{
		v1 = XMVectorDivide(v1, v0);

		// set w = 0 as tangent are pure direction
		v1 = XMVectorSelect(v1, XMVectorZero(), g_XMIdentityR3.v);
	}

	return v1;
}

XMVECTOR SpaceCurve::tangent(float t) const
{
	float l = length();
	if (m_isClose)
	{
		t = fmodf(t, l);
		if (t < .0f)
			t += l;
	}
	else
	{
		// clamp to [0,l]
		t = std::max(.0f, std::min(t, l));
	}

	// binary search for parameter
	int a = 0, b = m_anchors.size() - 1;

	while (b - a > 1)
	{
		unsigned int k = (a + b) / 2;
		if (m_anchors[k].w > t) b = k;
		else a = k;
	}

	XMVECTOR v0 = tangent(a);
	XMVECTOR v1 = tangent(b);
	float rt = (t - m_anchors[a].w) / (m_anchors[b].w - m_anchors[a].w);
	XMVECTOR v2 = XMVectorLerp(v0, v1, rt);

	return v2;
}

void SpaceCurve::updateLength()
{
	m_anchors[0].w = 0;
	XMVECTOR p0 = XMLoadFloat4A(&m_anchors[0]);
	XMVECTOR p1;
	for (int i = 1; i < m_anchors.size(); i++)
	{
		p1 = XMLoadFloat4A(&m_anchors[i]);
		float dt = XMVectorGetX(XMVector3Length(p1 - p0));
		m_anchors[i].w = m_anchors[i - 1].w + dt;
		p0 = p1;
	}
}