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

	template <class _Ty, class _Alloc = std::allocator<_Ty>>
	class BufferedStreamViewer
	{
	public:
		typedef const _Ty* const_pointer;
		typedef _Ty* pointer;
		typedef const _Ty& const_reference;
		typedef _Ty &  reference;
		typedef _Ty && rvalue_reference;

		//enum StreamViewMode
		//{
		//	Latest = 0,
		//	SequenceCaching = 1,
		//}

		BufferedStreamViewer(size_t BackBufferSize = 30U)
			: m_paused(false), m_Capicity(BackBufferSize)
		{
		}

		BufferedStreamViewer(const BufferedStreamViewer& rhs)
		{
			m_paused = rhs.m_paused;
			std::lock_guard<std::mutex> guard(rhs.m_BufferMutex);
			m_StreamingBuffer = rhs.m_StreamingBuffer;
		}

		BufferedStreamViewer(BufferedStreamViewer&& rhs) = default;

		~BufferedStreamViewer(void)
		{
			m_paused = true;
		}

		void Push(const_reference frame)
		{
			if (m_paused) return;

			std::lock_guard<std::mutex> guard(m_BufferMutex);
			if (m_StreamingBuffer.size() >= m_StreamingBuffer.max_size())
				m_StreamingBuffer.pop_front();
			m_StreamingBuffer.emplace_back(frame);
		}

		void Push(rvalue_reference frame)
		{
			if (m_paused) return;

			std::lock_guard<std::mutex> guard(m_BufferMutex);
			if (m_StreamingBuffer.size() >= m_Capicity)
				m_StreamingBuffer.pop_front();
			m_StreamingBuffer.emplace_back(std::move(frame));
		}

		pointer GetCurrent() const
		{
			return &m_ReadingBuffer;
		}

		pointer Peek(int idx = 0) const
		{
			if (idx == 0)
				return &m_ReadingBuffer;
			else if (idx - 1 < m_StreamingBuffer.size())
				return &m_StreamingBuffer[idx - 1];
			else return nullptr;
		}

		pointer PeekLatest() const
		{
			if (m_StreamingBuffer.empty())
				return &m_ReadingBuffer;
			return &m_StreamingBuffer.back();
		}

		bool MoveNext()
		{
			if (m_StreamingBuffer.empty()) return false;

			std::lock_guard<std::mutex> guard(m_BufferMutex);

			m_ReadingBuffer = m_StreamingBuffer.front();
			m_StreamingBuffer.pop_front();

			return true;
		}

		int MoveToLatest()
		{
			if (m_StreamingBuffer.empty()) return 0;

			std::lock_guard<std::mutex> guard(m_BufferMutex);

			m_ReadingBuffer = m_StreamingBuffer.back();

			int jump = m_StreamingBuffer.size();
			m_StreamingBuffer.clear();

			return jump;
		}

		std::deque<_Ty>& LockBuffer()
		{
			m_BufferMutex.lock();
			return m_StreamingBuffer;
		}

		void UnlockBuffer()
		{
			m_BufferMutex.unlock();
		}

		bool Empty() const
		{
			return m_StreamingBuffer.empty();
		}

		void Pause(bool pause)
		{
			m_paused = pause;
		}

	private:
		bool						m_paused;
		size_t						m_Capicity;
		mutable _Ty					m_ReadingBuffer;
		mutable std::deque<_Ty, _Alloc > m_StreamingBuffer;
		mutable std::mutex			m_BufferMutex;
	};

	typedef Causality::ArmatureFrame BodyFrame;

	// TrackedBody will host frames streaming from the Sensor
	// You can select to process all frams incoming or the lastest one only
	class TrackedBody : public IArmatureStreamAnimation
	{
	public:
		typedef Causality::ArmatureFrame		FrameType;

		typedef LowPassDynamicFilter<Vector3, float>	Vector3DynamicFilter;
		typedef LowPassDynamicFilter<Quaternion, float> QuaternionDynamicFilter;

		// Allows KinectSensor to modify
	private:
		friend Devices::KinectSensor;

		int						RefCount;
		uint64_t				Id;
		bool					m_IsTracked;
		time_t					m_LastTrackedTime;
		int						m_LostFrameCount;
		float					m_Distance;
		StaticArmature			m_armature;
		Devices::KinectSensor	*m_pKinectSensor;

		// Current pose data
		BufferedStreamViewer<FrameType>		m_FrameBuffer;

		// Hand States
		std::array<HandState, 2>			m_HandStates;

		void PushFrame(FrameType && frame);

	public:
		TrackedBody(size_t bufferSize = 30U);
		TrackedBody(const TrackedBody &) = default;
		TrackedBody(TrackedBody &&) = default;

		bool operator==(const TrackedBody& rhs) const { return Id == rhs.Id; }
		bool operator!=(const TrackedBody& rhs) const { return !(Id == rhs.Id); }

		void SetBufferSize(size_t pastFrame, size_t unprocessFrame);

		// Advance the stream to latest and peek the frame
		bool ReadLatestFrame() override;

		// Advance the stream by 1 and peek the frame
		bool ReadNextFrame() override;

		// Peek the current frame head
		const FrameType& PeekFrame() const override;

		const IArmature& GetArmature() const override;

		bool IsAvailable() const override;
		// Get Historical or Future 
		// relativeIndex = 0, returns latest unprocess frame
		// relativeIndex > 0, return "future" frames
		// relativeIndex < 0, return past frames
		bool GetFrameAt(FrameType& outFrame, int relativeIndex) const;

		// Get Historical or Future frames
		// Body-pose is Lerp-ed between frame
		bool GetFrameAt(FrameType& outFrame, const time_seconds& relateTimve) const;

		// Accessers
		Devices::KinectSensor& GetSensor() const { return *m_pKinectSensor; }
		time_t	  GetLastTrackedTime() const { return m_LastTrackedTime; }
		uint64_t  GetTrackId() const { return Id; }
		HandState GetHandState(HandType hand) const { return m_HandStates[hand]; }
		bool	  IsTracked() const { return m_IsTracked; }
		float	  DistanceToSensor() const;

		void	  SetArmatureProportion(ArmatureFrameConstView frameView);
		void	  FireFrameArrivedForLatest();
		// Events
		Event<const TrackedBody&, const FrameType&>	OnFrameArrived;
	public:
		// Skeleton Structure and basic body parameter
		// Shared through all players since they have same structure
		static std::unique_ptr<StaticArmature> BodyArmature;

		void AddRef();
		void Release();
	};

	class ITrackedBodySelector abstract
	{
	public:
		virtual TrackedBody* ReselectBody() = 0;
	};

	namespace Devices
	{
		// An aggregate of Kinect Resources
		class KinectSensor : public std::enable_shared_from_this<KinectSensor>
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

			bool Start();
			void Stop();
			bool Pause();
			bool Resume();

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


