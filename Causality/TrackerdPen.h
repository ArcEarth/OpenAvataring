#pragma once
#include "TrackedObjectControl.h"

namespace Causality
{
	class KeyboardMouseFirstPersonControl;

	class TrackedPen : public TrackedObjectControl
	{
	public:
		TrackedPen();

		~TrackedPen();

		void OnParentChanged(SceneObject* oldParent) override;

		void Update(const time_seconds& time_delta) override;

		bool UpdateFromLeapFinger(double dt);

		bool IsInking() const;
		bool IsErasing() const;
		bool IsDraging() const;

		bool IsVisible() const;

		void setMouse(const KeyboardMouseFirstPersonControl* m);

		// = 0.0145 m
		static const float TipLength;
	protected:
		bool  m_visible;
		float m_inkingStr;
		float m_erasingStr;
		float m_dragingStr;

		const KeyboardMouseFirstPersonControl*
			m_mouse;
	};
}