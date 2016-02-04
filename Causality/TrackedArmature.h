#pragma once

#include <atomic>
#include "BufferedStreamViewer.h"
#include "Animations.h"
#include "Events.h"

namespace Causality
{
	// interface for tracking objects
	class TrackedObject
	{
	public:
		TrackedObject();
		uint64_t  GetTrackId() const { return m_id; }
		time_t	  GetLastTrackedTime() const { return m_lastTrackedTime; }
		bool	  IsTracked() const { return m_isTracked; }

		int  RefCount() const { return m_refCount; }
		void AddRef();
		void Release();

		int  GetLostFrameCount() const { return m_lostFrameCount; }
		void ResetLostFrameCount() { m_lostFrameCount = 0; }
		int  IncreaseLostFrameCount(int frames = 1) { m_lostFrameCount += frames; return m_lostFrameCount; }

		bool operator==(const TrackedObject& rhs) const { return m_id == rhs.m_id; }
		bool operator!=(const TrackedObject& rhs) const { return m_id != rhs.m_id; }

		bool operator<(const TrackedObject& rhs) const { return m_id < rhs.m_id; }
		bool operator>(const TrackedObject& rhs) const { return m_id > rhs.m_id; }
		bool operator<=(const TrackedObject& rhs) const { return m_id <= rhs.m_id; }
		bool operator>=(const TrackedObject& rhs) const { return m_id >= rhs.m_id; }

		bool operator==(uint64_t id) const { return m_id == id; }
		bool operator!=(uint64_t id) const { return m_id != id; }

		bool operator<(uint64_t id) const { return m_id < id; }
		bool operator>(uint64_t id) const { return m_id > id; }
		bool operator<=(uint64_t id) const { return m_id <= id; }
		bool operator>=(uint64_t id) const { return m_id >= id; }

		// The event that signals this object is lost and will be destroy
		Event<const TrackedObject&> Lost;
	protected:
		uint64_t				m_id;
		std::atomic_int			m_refCount;
		std::atomic_bool		m_isTracked;
		time_t					m_lastTrackedTime;
		int                     m_lostFrameCount;
	};

   	// An interface for tracking hireachical structure
	class TrackedArmature : public IArmatureStreamAnimation, public TrackedObject
	{
	public:
		typedef ArmatureFrame		FrameType;
		TrackedArmature(size_t bufferSize = 32U);
        TrackedArmature(const IArmature& armature, size_t bufferSize = 32U);
        
		void SetArmature(const IArmature& armature);
		const IArmature& GetArmature() const override;
		// Advance the stream by 1 
		bool ReadNextFrame() override;
		// Advance the stream to latest
		bool ReadLatestFrame() override;
		// Peek the current frame head
		const FrameType& PeekFrame() const override;
		bool IsAvailable() const override;

		bool GetFrameAt(FrameType& outFrame, int relativeIndex) const;
		bool GetFrameAt(FrameType& outFrame, const time_seconds& relateTimve) const;

		void PushFrame(time_t time, FrameType && frame);
		void FireFrameArrivedForLatest();
		void SetArmatureProportion(ArmatureFrameConstView frameView);
	protected:
		StaticArmature			m_armature;

		// Current pose data
		BufferedStreamViewer<FrameType>		
								m_frameBuffer;

	public:
		Event<const TrackedArmature&, const FrameType&>	OnFrameArrived;
	};
 
}