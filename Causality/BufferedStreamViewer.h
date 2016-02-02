#pragma once
#include <mutex>
#include <deque>

namespace Causality
{
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

		BufferedStreamViewer(size_t BackBufferSize)
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
}