#pragma once
#include "PlayerSelector.h"

namespace Causality
{
	class LeapPlayerSelector : public IPlayerSelector
	{
	public:
		enum SelectionMode
		{
			None = 0,
			Sticky = 1,
			Closest = 2,
			ClosestStickly = 3,
			PreferLeft = 4,
			PreferRight = 8,
			JoinMuiltiplePlayer = 16, // Merge all player into one skeleton which connects all the hip centers
		};

	private:
		typedef std::function<void(const IArmatureStreamAnimation&, const IArmatureStreamAnimation::FrameType&)> FrameEventFunctionType;
		typedef std::function<void(IArmatureStreamAnimation*, IArmatureStreamAnimation*)> PlayerEventFunctionType;
		FrameEventFunctionType	fpFrameArrived;
		PlayerEventFunctionType	fpTrackedBodyChanged;

		TrackedBody*						pCurrent;
		shared_ptr<Devices::KinectSensor>	pKinect;

		SelectionMode			mode;
		EventConnection			con_tracked;
		EventConnection			con_lost;
		EventConnection			con_frame;

	public:
		explicit KinectPlayerSelector(Devices::KinectSensor* pKinect, SelectionMode mode = Sticky);
		~KinectPlayerSelector();
		void Reset();
		void Initialize(Devices::KinectSensor* pKinect, SelectionMode mode = Sticky);
		void ChangePlayer(TrackedBody* pNewPlayer);

		using IPlayerSelector::operator bool;
		using IPlayerSelector::operator*;
		using IPlayerSelector::operator->;
		using IPlayerSelector::operator==;
		using IPlayerSelector::operator!=;

		void SetFrameCallback(const FrameEventFunctionType& callback) override;
		void SetPlayerChangeCallback(const PlayerEventFunctionType& callback) override;
		IArmatureStreamAnimation* Get() override;

		void OnPlayerTracked(TrackedBody& body);
		void OnPlayerLost(TrackedBody& body);
		void ReSelectFromAllTrackedBodies();

		void ChangeSelectionMode(SelectionMode mdoe);

		SelectionMode CurrentSelectionMode() const
		{
			return mode;
		}

	};
}