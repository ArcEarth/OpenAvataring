#pragma once
#include <Causality\Events.h>
#include <Causality\SmartPointers.h>
#include <Causality\Animations.h>

namespace Causality
{
	class TrackedBody;
	namespace Devices
	{
		class KinectSensor;
	}

	class IPlayerSelector abstract
	{
	public:
		typedef std::function<void(const IArmatureStreamAnimation&, const IArmatureStreamAnimation::FrameType&)> FrameEventFunctionType;
		typedef std::function<void(IArmatureStreamAnimation*, IArmatureStreamAnimation*)> PlayerEventFunctionType;

		virtual void SetFrameCallback(const FrameEventFunctionType& callback) = 0;
		virtual void SetPlayerChangeCallback(const PlayerEventFunctionType& callback) = 0;

		virtual IArmatureStreamAnimation* Get() = 0;
		const IArmatureStreamAnimation* Get() const { return const_cast<IPlayerSelector*>(this)->Get(); }

		IArmatureStreamAnimation* operator->()
		{
			return Get();
		}

		const IArmatureStreamAnimation* operator->() const
		{
			return Get();
		}

		IArmatureStreamAnimation& operator*()
		{
			return *Get();
		}

		const IArmatureStreamAnimation& operator*() const
		{
			*Get();
		}

		operator bool() const { return Get() != nullptr; }
		bool operator == (nullptr_t) const { return Get() == nullptr; }
		bool operator != (nullptr_t) const { return Get() != nullptr; }
	};

	/// <summary>
	/// Helper class for Selecting sensor tracked bodies.
	/// Specify the behavier by seting the Selection Mode.
	/// Act as a Smart pointer to the actual body.
	/// Provide callback for notifying selected body changed and recieved a frame.
	/// </summary>
	class KinectPlayerSelector : public IPlayerSelector
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