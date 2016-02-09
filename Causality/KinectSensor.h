#ifndef	KINECT_2_USER
#define KINECT_2_USER

#ifdef _MSC_VER
#pragma once
#endif  // _MSC_VER
#include "BCL.h"
#include <wrl\client.h>
#include <memory>
#include <mutex>
#include "DirectXMathExtend.h"
#include "Armature.h"
#include "Animations.h"
#include "Common\Filter.h"
#include "Events.h"
#include "TrackedArmature.h"
#include "StreamDevice.h"

namespace Causality
{
	namespace Devices
	{
		class KinectSensor;
	}

#ifndef _HandType_
#define _HandType_
	typedef enum _HandType HandType;


	enum _HandType
	{
		HandType_NONE = 0,
		HandType_LEFT = (HandType_NONE + 1),
		HandType_RIGHT = (HandType_LEFT + 1)
	};
#endif // _HandType_
#ifndef _HandState_
#define _HandState_
	typedef enum _HandState HandState;


	enum _HandState
	{
		HandState_Unknown = 0,
		HandState_NotTracked = 1,
		HandState_Open = 2,
		HandState_Closed = 3,
		HandState_Lasso = 4
	};
#endif // _HandState_

#ifndef _JointType_
#define _JointType_
	typedef enum _JointType JointType;


	enum _JointType
	{
		JointType_SpineBase = 0,
		JointType_SpineMid = 1,
		JointType_Neck = 2,
		JointType_Head = 3,
		JointType_ShoulderLeft = 4,
		JointType_ElbowLeft = 5,
		JointType_WristLeft = 6,
		JointType_HandLeft = 7,
		JointType_ShoulderRight = 8,
		JointType_ElbowRight = 9,
		JointType_WristRight = 10,
		JointType_HandRight = 11,
		JointType_HipLeft = 12,
		JointType_KneeLeft = 13,
		JointType_AnkleLeft = 14,
		JointType_FootLeft = 15,
		JointType_HipRight = 16,
		JointType_KneeRight = 17,
		JointType_AnkleRight = 18,
		JointType_FootRight = 19,
		JointType_SpineShoulder = 20,
		JointType_HandTipLeft = 21,
		JointType_ThumbLeft = 22,
		JointType_HandTipRight = 23,
		JointType_ThumbRight = 24,
		JointType_Count = (JointType_ThumbRight + 1)
	};
#endif // _JointType_

	typedef Causality::ArmatureFrame BodyFrame;

	// TrackedBody will host frames streaming from the Sensor
	// You can select to process all frams incoming or the lastest one only
	class TrackedBody : public TrackedArmature
	{
	public:
		typedef Causality::ArmatureFrame		FrameType;

		typedef LowPassDynamicFilter<Vector3, float>	Vector3DynamicFilter;
		typedef LowPassDynamicFilter<Quaternion, float> QuaternionDynamicFilter;

		// Allows KinectSensor to modify
	private:
		friend Devices::KinectSensor;

		float					 m_Distance;
		Devices::KinectSensor*   m_pKinectSensor;

		// Hand States
		std::array<HandState, 2> m_HandStates;

	public:
		TrackedBody(size_t bufferSize = 32U);
		TrackedBody(const TrackedBody &) = delete;
		TrackedBody(TrackedBody &&) = default;

		void SetBufferSize(size_t pastFrame, size_t unprocessFrame);

		Devices::KinectSensor& GetSensor() const { return *m_pKinectSensor; }
		HandState GetHandState(HandType hand) const { return m_HandStates[hand]; }
		float	  DistanceToSensor() const { return m_Distance; }

	public:
		// Skeleton Structure and basic body parameter
		// Shared through all players since they have same structure
		static std::unique_ptr<StaticArmature> BodyArmature;
	};

	class ITrackedBodySelector abstract
	{
	public:
		virtual TrackedBody* ReselectBody() = 0;
	};

	namespace Devices
	{
		// An aggregate of Kinect Resources
		class KinectSensor : public std::enable_shared_from_this<KinectSensor>, public IStreamDevice
		{
		public:
			typedef std::weak_ptr<KinectSensor>		Weakptr;
			typedef std::shared_ptr<KinectSensor>	Refptr;

			static Weakptr	wpCurrentDevice;
			static Refptr	GetForCurrentView();

			const StaticArmature& Armature() const { return *TrackedBody::BodyArmature; }
			Refptr GetRef() { return shared_from_this(); }

			~KinectSensor();

			bool IsConnected() const;

			// Player Event event interface!
			Event<TrackedBody&> OnPlayerTracked;
			Event<TrackedBody&> OnPlayerLost;
			//Platform::Fundation::Event<const TrackedBody&> OnPlayerPoseChanged;
			//Platform::Fundation::Event<const TrackedBody&, HandEnum, HandState> OnPlayerHandStateChanged;

			// DeviceCoordinate Matrix will be use to transform all input data every frame
			// If Kinect is moving, this function should be called every frame
			// Should be valiad before the call to Process Frame
			void SetDeviceCoordinate(const DirectX::Matrix4x4& mat);
			void EnableMirrowing(bool enable_mirrow);
			bool IsMirrowingEnable() const;

			const DirectX::Matrix4x4& GetDeviceCoordinate();

			// Process frame and trigger the events
			void ProcessFrame();

			bool Start() override;
			void Stop() override;
			bool Pause();
			bool Resume();

			bool Initialize(const ParamArchive * archive) override;
			bool Update() override;
			bool IsStreaming() const override;
			bool IsAsychronize() const override;

			// Return the list of CURRENT Tracked bodies
			const std::list<TrackedBody> &GetTrackedBodies() const;
			std::list<TrackedBody> &GetTrackedBodies();

			// Static Constructors!!!
			static Refptr CreateDefault();

		public:
			// You should not use this
			KinectSensor();

		public:
			static JointType JointsParent[JointType_Count];

		private:

			class Impl;
			std::unique_ptr<Impl> pImpl;

		};

	}
}


#endif


