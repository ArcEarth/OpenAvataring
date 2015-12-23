#pragma once

#include <vector>
#include <memory>
#include "DirectXMathExtend.h"
#include <span.h>

namespace Geometrics
{
	using DirectX::Vector3;
	using DirectX::XMVECTOR;
	using DirectX::XMFLOAT4A;
	using DirectX::FXMVECTOR;

	template <class T>
	void laplacianSmooth(gsl::span<T> curve, float alpha/* = 0.8f*/, unsigned IterationTimes /*= 1*/, bool closeLoop /*=false*/)
	{
		unsigned int N = curve.size();
		std::vector<T> cache(N);
		T* BUFF[2] = { &curve[0],&cache[0] };
		unsigned int src = 0, dst = 1;
		float invAlpha = 0.5f * (1.0f - alpha);

		for (unsigned int k = 0; k < IterationTimes; k++)
		{
			if (!closeLoop)
			{
				BUFF[dst][0] = BUFF[src][0];
				BUFF[dst][N - 1] = BUFF[src][N - 1];
			}
			else
			{
				BUFF[dst][0] = alpha * BUFF[src][0] + invAlpha * (BUFF[src][N - 1] + BUFF[src][1]);
				BUFF[dst][N - 1] = alpha * BUFF[src][0] + invAlpha * (BUFF[src][N - 2] + BUFF[src][0]);
			}

			for (unsigned int i = 1; i < N - 1; i++)
			{
				BUFF[dst][i] = alpha * BUFF[src][i] + invAlpha * (BUFF[src][i - 1] + BUFF[src][i + 1]);
			}
			dst = !dst;
			src = !src;
		}
		if (dst == 0)
		{
			for (unsigned int i = 0; i < N; i++)
			{
				BUFF[0][i] = BUFF[1][i];
			}
		}
	}

	// Class to represent a spatial curve with anchor points
	// Provide method for linear sampling from it
	class SpaceCurve
	{
	public:
		typedef std::vector<XMFLOAT4A, DirectX::AlignedAllocator<XMFLOAT4A>> AnchorCollection;
		SpaceCurve();
		~SpaceCurve();
		explicit SpaceCurve(const gsl::span<Vector3>& trajectory, bool closeLoop = false);

		bool isClose() const { return m_isClose; }
		void setClose(bool close);
		void closeLoop() { setClose(true); }

		size_t size() const;
		bool empty() const { return m_anchors.empty(); }
		float length() const;
		XMFLOAT4A* data() { return m_anchors.data(); }
		const XMFLOAT4A* data() const { return m_anchors.data(); }
		void clear() { m_anchors.clear(); }
		bool push_back(const Vector3& p, bool force = false);
		bool XM_CALLCONV push_back(FXMVECTOR p, bool force = false);
		bool XM_CALLCONV append(FXMVECTOR p, bool force = false) { return push_back(p,force); }
		XMVECTOR back() const;

		// Retrive the point at parameter position 't' belongs to [0,1]
		// This O(LogN) level operation
		// perform a binary search in anchors 
		// returns a 4D vector where xyz represent the Position 
		// w is the length from start to point;
		XMVECTOR extract(float t) const;
		XMVECTOR position(float t) const { return extract(t); }
		XMVECTOR position(int idx) const {
			using namespace DirectX;
			XMVECTOR v = XMLoadFloat4A(&m_anchors[idx]); 
			v = XMVectorSelect(v, g_XMIdentityR3.v, g_XMIdentityR3.v);
			return v;
		}
		// eval tangent at parameter position t
		XMVECTOR tangent(float t) const;
		// eval tangent at anchor position idx
		XMVECTOR tangent(int idx) const;

		inline XMVECTOR operator[](int idx) const { return XMLoadFloat4A(&m_anchors[idx]); }
		inline XMVECTOR operator()(float t) const { return extract(t); }


		std::vector<Vector3> sample(size_t sampleCount) const;
		void resample(size_t anchorCount, bool smooth = true);

		void smooth(float alpha/* = 0.8f*/, unsigned iteration /*= 1*/);

		std::vector<Vector3> FixIntervalSampling(float Interval) const;
		std::vector<Vector3> FixCountSampling(unsigned int SampleSegmentCount, bool Smooth = true) const;
		std::vector<Vector3> FixCountSampling2(unsigned int SampleSegmentCount) const;

		void updateLength();

		const XMFLOAT4A& anchor(int idx) const
		{
			return m_anchors[idx];
		}

		XMFLOAT4A& anchor(int idx)
		{
			return m_anchors[idx];
		}

		const AnchorCollection& anchors() const
		{
			return m_anchors;
		}

	private:
		// The data we stored is actually aligned on 16-byte boundary , so , use it as a XMFLOAT4A
		AnchorCollection m_anchors;
		bool m_isClose;
	};

}