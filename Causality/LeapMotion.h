#pragma once

#include <atomic>

#include "Math3D.h"
#include "Events.h"
#include "SmartPointers.h"
#include "StreamDevice.h"
#include "TrackedArmature.h"

namespace Leap
{
	class Controller;
}

namespace Causality
{
	namespace Devices
	{
		class LeapSensor;
		namespace Internal
		{
			class LeapListener;
		}
	}

	// https://developer.leapmotion.com/documentation/cpp/devguide/Leap_Overview.html
	// Note the bone for Thumb are different in anotomic defination
	// https://commons.wikimedia.org/wiki/File:Scheme_human_hand_bones-en.svg
	class TrackedHand : public TrackedArmature
	{
	public:
		enum JointType : int
		{
			JointType_Null = -1,
			JointType_Wrist = 0,
			JointType_ThumbMetacarpal,
			JointType_ThumbProximal,
			JointType_ThumbIntermediate,
			JointType_ThumbDistal,
			JointType_IndexMetacarpal,
			JointType_IndexProximal,
			JointType_IndexIntermediate,
			JointType_IndexDistal,
			JointType_MiddleMetacarpal,
			JointType_MiddleProximal,
			JointType_MiddleIntermediate,
			JointType_MiddleDistal,
			JointType_RingMetacarpal,
			JointType_RingProximal,
			JointType_RingIntermediate,
			JointType_RingDistal,
			JointType_PinkyMetacarpal,
			JointType_PinkyProximal,
			JointType_PinkyIntermediate,
			JointType_PinkyDistal,
			JointType_Count
		};

		friend Devices::LeapSensor;
		friend Devices::Internal::LeapListener;

		TrackedHand(int64_t id, bool isLeft);

		bool IsRight() const;
		bool IsLeft() const;

		Devices::LeapSensor& GetSensor() const { return *m_pSensor; }

	private:
		Devices::LeapSensor	*m_pSensor;
		bool m_isLeft;

	private:
		// Default hand armature
		static unique_ptr<StaticArmature> s_pHandArmature;
		static int s_HandArmatureParentMap[TrackedHand::JointType_Count];
		static void IntializeHandArmature();
	};

	namespace Devices
	{
		class LeapSensor : public std::enable_shared_from_this<LeapSensor>, public IStreamDevice
		{
		public:
			static std::weak_ptr<LeapSensor> wpCurrentDevice;
			static std::shared_ptr<LeapSensor> GetForCurrentView();

			LeapSensor();
			LeapSensor(bool useEvent , bool isFixed );
			~LeapSensor();

			void Initialize(bool useEvent, bool isFixed);

			Leap::Controller& Controller();
			const Leap::Controller& Controller() const;


			//void SetMotionProvider(DirectX::ILocatable* pHeadLoc, DirectX::IOriented *pHeadOrient);
			// Assign Leap motion's coordinate in world space
			void SetDeviceWorldCoord(const Matrix4x4 &m);

			// The transform matrix convert Leap Coordinate to World coordinate
			XMMATRIX ToWorldTransform() const;

			Event<LeapSensor &>		Connected;
			Event<LeapSensor &>		Disconnected;

			Event<TrackedHand &>	HandTracked;
			Event<TrackedHand &>	HandLost;

			// Inherited via IStreamDevice
			virtual bool Initialize(const ParamArchive * archive) override;
			virtual bool Start() override;
			virtual void Stop() override;
			// Synchronize update method (pull frame)
			// To use with seqential logic and distribute events
			virtual bool Update() override;
			virtual bool IsStreaming() const override;
			virtual bool IsAsychronize() const override;

			bool Pause();
			bool Resume();

			const std::list<TrackedHand> &GetTrackedHands() const;
			std::list<TrackedHand>		 &GetTrackedHands();

		private:
			bool			 m_useEvent;
			bool			 m_isFixed;
			bool			 m_started;
			std::atomic_bool m_connected;

			std::unique_ptr<Internal::LeapListener>
				pListener;
			std::unique_ptr<Leap::Controller>
				pController;
		};
	}
}