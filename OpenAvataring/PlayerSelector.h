#pragma once
#include <Causality\Events.h>
#include <Causality\SmartPointers.h>
#include <Causality\Animations.h>

namespace Causality
{
	class TrackedArmature;

	namespace Devices
	{
		class IStreamDevice;
		class KinectSensor;
		class LeapSensor;
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

	class PlayerSelectorBase : public IPlayerSelector
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
			MergeAll = 16, // Merge all player into one skeleton which connects all the hip centers
		};

	protected:
		class MergedTrackedArmature;

		FrameEventFunctionType	fpFrameArrived;
		PlayerEventFunctionType	fpTrackedBodyChanged;

		TrackedArmature*					m_current;
		shared_ptr<Devices::IStreamDevice>	m_sensor;
		unique_ptr<MergedTrackedArmature>	m_merged;
		unique_ptr<MergedTrackedArmature>	m_oldmerged;
		std::vector<TrackedArmature*>		m_candidates;

		SelectionMode						mode;
		EventConnection						con_tracked;
		EventConnection						con_lost;
		EventConnection						con_frame;


		PlayerSelectorBase();
	public:
		using IPlayerSelector::operator bool;
		using IPlayerSelector::operator*;
		using IPlayerSelector::operator->;
		using IPlayerSelector::operator==;
		using IPlayerSelector::operator!=;

		~PlayerSelectorBase();

		// Only reason this is virtual is we need the interface to access TrackedArmature List from the streaming device, which is not converged yet
		virtual void GetTrackedArmatures(std::vector<TrackedArmature*> &armatures) = 0;
		virtual float Distance(const TrackedArmature& body) const;

		void Reset();
		void ChangePlayer(TrackedArmature* pNewPlayer);
		void SetFrameCallback(const FrameEventFunctionType& callback) override;
		void SetPlayerChangeCallback(const PlayerEventFunctionType& callback) override;
		IArmatureStreamAnimation* Get() override;
		void OnPlayerTracked(TrackedArmature& body);
		void OnPlayerLost(TrackedArmature& body);
		void Reselect();

		void ChangeSelectionMode(SelectionMode mdoe);

		SelectionMode CurrentSelectionMode() const
		{
			return mode;
		}
	};

	/// <summary>
	/// Helper class for Selecting sensor tracked bodies.
	/// Specify the behavier by seting the Selection Mode.
	/// Act as a Smart pointer to the actual body.
	/// Provide callback for notifying selected body changed and recieved a frame.
	/// </summary>
	class KinectPlayerSelector : public PlayerSelectorBase
	{
	public:
		explicit KinectPlayerSelector(Devices::KinectSensor* pKinect, SelectionMode mode = Sticky);
		~KinectPlayerSelector();
		void Initialize(Devices::KinectSensor* pKinect, SelectionMode mode = Sticky);
		void GetTrackedArmatures(std::vector<TrackedArmature*> &armatures) override;
		float Distance(const TrackedArmature& body) const override;
	};

	class LeapPlayerSelector : public PlayerSelectorBase
	{
	public:
		explicit LeapPlayerSelector(Devices::LeapSensor* pLeap, SelectionMode mode = Sticky);
		~LeapPlayerSelector();
		void Initialize(Devices::LeapSensor* pLeap, SelectionMode mode = Sticky);
		void GetTrackedArmatures(std::vector<TrackedArmature*> &armatures) override;
		float Distance(const TrackedArmature& body) const override;
	};
}