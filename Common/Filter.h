#pragma once

#include <cmath>
#include <algorithm>

namespace Causality
{

	//////////////////////////////////////////////////////////////////////
	//
	// Filter 
	// 
	//////////////////////////////////////////////////////////////////////
	// Template class for filtering
	//  
	template <class TValue, class TScaler = double>
	class Filter
	{
	public:
		//////////////////////////////////////////////////////////////////////
		// UpdateFrequency (Hz) 
		Filter(TScaler* pUpdateFrequency)
		{
			m_pUpdateFrequency = pUpdateFrequency;
			m_FirstTime = true;
		}

		Filter() = default;

		virtual const TValue& Apply(const TValue &value) = 0;

		const TValue& Delta() const { return m_Delta; }
		const TValue& Value() const { return m_PrevValue; }

		virtual ~Filter(void) { }

		// Reset the filter to it's initial state
		void Reset() { m_FirstTime = true; };
		void SetUpdateFrequency(TScaler* updateFrequency) { m_pUpdateFrequency = updateFrequency; };
		TScaler GetUpdateFrequency() const { return *m_pUpdateFrequency; }


	protected:

		TScaler* m_pUpdateFrequency;

		TValue m_PrevValue;
		TValue m_Delta;
		bool m_FirstTime;

	};

	template <class _Ty, class _TScalar>
	struct default_filter_operators
	{
		typedef _Ty Ty;
		typedef _TScalar TScalar;

		static _Ty lerp(const _Ty& lhs, const _Ty&rhs, _TScalar t)
		{
			return lhs + (rhs - lhs) * t;
		}

		static _Ty scale(const _Ty& lhs, _TScalar s)
		{
			return lhs * s;
		}

		// difference From lhs To rhs
		static _Ty difference(const _Ty& lhs, const _Ty&rhs)
		{
			return rhs - lhs;
		}

		static _TScalar norm(const _Ty& lhs)
		{
			return static_cast<_TScalar>(lhs);
		}

	};

	template <class _Ty, class _TScalar>
	struct filter_operators : default_filter_operators<_Ty, _TScalar>
	{
	};

#ifdef DIRECTX_MATH_VERSION
	template <>
	struct filter_operators<DirectX::Quaternion, float>
	{
		using _Ty = DirectX::Quaternion;
		using _TScaler = float;
		static _Ty lerp(const _Ty& lhs, const _Ty&rhs, _TScaler t)
		{
			return DirectX::Quaternion::Slerp(lhs, rhs, t);
		}

		// Maybe using Axis-Angle rep are better?
		static _Ty scale(const _Ty& lhs, _TScaler s)
		{
			using namespace DirectX;
			XMVECTOR lq = XMQuaternionLn(lhs);
			lq *= s;
			lq = XMQuaternionExp(lq);
			return lq;
		}

		static _Ty difference(const _Ty& lhs, const _Ty&rhs)
		{
			using namespace DirectX;
			XMVECTOR lq = XMLoad(lhs);
			XMVECTOR rq = XMLoad(rhs);
			lq = XMQuaternionInverse(lhs);
			lq = XMQuaternionMultiply(lq, rq);
			return lq;
		}

		static _TScaler norm(const _Ty& lhs)
		{
			using namespace DirectX;
			XMVECTOR lq = XMQuaternionLn(lhs);
			lq = XMVector3Length(lq);
			return XMVectorGetX(lq);
		}
	};
	template <>
	struct filter_operators<DirectX::Vector2, float> : default_filter_operators<DirectX::Vector2, float>
	{
		static float norm(const Ty& lhs)
		{
			return lhs.Length();
		}
	};
	template <>
	struct filter_operators<DirectX::Vector3, float> : default_filter_operators<DirectX::Vector3, float>
	{
		static float norm(const Ty& lhs)
		{
			return lhs.Length();
		}
	};
	template <>
	struct filter_operators<DirectX::Vector4, float> : default_filter_operators<DirectX::Vector4, float>
	{
		static float norm(const Ty& lhs)
		{
			return lhs.Length();
		}
	};
#endif


	//////////////////////////////////////////////////////////////////////
	//
	// Low Pass Filter 
	// 
	//////////////////////////////////////////////////////////////////////
	// set the following before using: 
	//		mCutoffFrequency
	//

	template <class TValue, class TScaler = double, class TFilterOperator = filter_operators<TValue, TScaler>>
	class LowPassFilter
		: public Filter<TValue, TScaler>
	{
	public:
		LowPassFilter(TScaler* pUpdateFrequency, TScaler cutOffFrequency = 0)
			: Filter<TValue, TScaler>(pUpdateFrequency), m_CutoffFrequency(cutOffFrequency) {};

		LowPassFilter() = default;

		virtual const TValue& Apply(const TValue &value) override
		{
			/*

			Let's say Pnf the filtered position, Pn the non filtered position and Pn-1f the previous filtered position,
			Te the sampling period (in second) and tau a time constant calculated from the cut-off frequency fc.

			tau = 1 / (2 * pi * fc)
			Pnf = ( Pn + tau/Te * Pn-1f ) * 1/(1+ tau/Te)

			Attention: tau >= 10 * Te
			*/

			if (m_FirstTime)
			{
				m_PrevValue = value;
				m_Delta = TFilterOperator::difference(m_PrevValue, m_PrevValue);
				m_FirstTime = false;
				return m_PrevValue;
			}

			const TScaler one = TScaler(1), zero = TScaler(0), twopi = TScaler(2 * 3.14159265);

			TScaler updateFrequency = *m_pUpdateFrequency;

			TScaler Te(one / updateFrequency);		// the sampling period (in seconds)
			TScaler Tau(one / (twopi * m_CutoffFrequency));	// a time constant calculated from the cut-off frequency

			auto t = one / (one + (Tau / Te));

			//m_Delta = t * (value - m_PrevValue);

			//m_PrevValue += m_Delta;
			m_Delta = TFilterOperator::lerp(m_PrevValue, value, t);

			m_PrevValue = TFilterOperator::difference(m_PrevValue, m_Delta);

			std::swap(m_Delta, m_PrevValue);

			return m_PrevValue;
		}

		void SetCutoffFrequency(TScaler f) { m_CutoffFrequency = f; };

		TScaler GetCutoffFrequency(void) { return m_CutoffFrequency; };

	protected:
		TScaler					m_CutoffFrequency;
	};

	//////////////////////////////////////////////////////////////////////
	//
	// Low Pass Dynamic (or Adjustable) Filter 
	// 
	//////////////////////////////////////////////////////////////////////
	// adjusts the cutoff filter depending on the velocity
	// set the following before using: 
	//		mUpdateFrequency (Hz) 
	//		mCutoffFrequency the lower cuttoff frequency
	//		mCutoffFrequencyHigh
	//		mVelocityLow (mm/s) the speed at which mCutoffFrequencyLow is reached 
	//		mVelocityHigh (mm/s) the speed at which mCutoffFrequencyHigh is reached
	template <class TValue, class TScaler = double, class TFilterOperator = filter_operators<TValue, TScaler>>
	class LowPassDynamicFilter : public LowPassFilter<TValue, TScaler>
	{
	protected:
		LowPassFilter<TValue,TScaler>	m_VelocityFilter;
		TValue					m_LastPositionForVelocity;
		TScaler					m_CutoffFrequencyHigh, m_VelocityLow, m_VelocityHigh;

	public:
		LowPassDynamicFilter(TScaler* updateFrequency) : LowPassFilter<TValue, TScaler>(updateFrequency), m_VelocityFilter(updateFrequency) {  };
		LowPassDynamicFilter() = default;
		virtual const TValue& Apply(const TValue &value)
		{
			using namespace std;

			// special case if first time being used
			if (m_FirstTime)
			{
				m_PrevValue = value;
				m_LastPositionForVelocity = value;
				m_FirstTime = false;
			}


			TScaler updateFrequency = *m_pUpdateFrequency;

			// first get an estimate of velocity (with filter)
			TValue positionForVelocity = m_VelocityFilter.Apply(value);
			TValue vel = TFilterOperator::difference(m_LastPositionForVelocity, positionForVelocity);
			TScaler velNorm = TFilterOperator::norm(vel) * updateFrequency;

			m_LastPositionForVelocity = positionForVelocity;

			// constants
			const TScaler one = TScaler(1), zero = TScaler(0), twopi = TScaler(2 * 3.14159265);
			// interpolate between frequencies depending on velocity

			TScaler t = (velNorm - m_VelocityLow) / (m_VelocityHigh - m_VelocityLow);
			t = min(max(t, zero), one);
			TScaler cutoff((m_CutoffFrequencyHigh * t) + (m_CutoffFrequency * (one - t)));
			TScaler Te(one / updateFrequency);		// the sampling period (in seconds)
			TScaler Tau(one / (twopi * cutoff));	// a time constant calculated from the cut-off frequency

			t = one / (one + (Tau / Te));

			m_Delta = TFilterOperator::lerp(m_PrevValue, value, t);

			m_PrevValue = TFilterOperator::difference(m_PrevValue, m_Delta);

			std::swap(m_Delta, m_PrevValue);

			return m_PrevValue;
		}

		// Parameter Properties
		TScaler GetCutoffFrequencyHigh() const { return m_CutoffFrequencyHigh; }
		TScaler GetCutoffFrequencyLow() const { return m_CutoffFrequency; }
		TScaler GetVelocityLow() const { return *m_VelocityLow; }
		TScaler GetVelocityHigh() const { return *m_VelocityHigh; }

		void SetCutoffFrequencyLow(TScaler f) { m_CutoffFrequency = f; SetCutoffFrequencyVelocity(); };
		void SetCutoffFrequencyHigh(TScaler f) { m_CutoffFrequencyHigh = f; SetCutoffFrequencyVelocity(); };
		void SetVelocityLow(TScaler f) { m_VelocityLow = f; };
		void SetVelocityHigh(TScaler f) { m_VelocityHigh = f; };

	protected:

		// cutoff freq for velocity
		// equal to mCutoffFrequency  + 0.75 * (mCutoffFrequencyHigh - mCutoffFrequency)
		void SetCutoffFrequencyVelocity()
		{
			m_VelocityFilter.SetCutoffFrequency((TScaler)(m_CutoffFrequency + 0.75 * (m_CutoffFrequencyHigh - m_CutoffFrequency)));
		}
	};

	//
	//////////////////////////////////////////////////////////////////////
	// 
}