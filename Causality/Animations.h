#pragma once
#include "Armature.h"
#include <vector>
#include <chrono>

namespace Causality
{
	using time_seconds = std::chrono::duration<double>;
	typedef time_seconds TimeScalarType;

	enum ANIM_STANDARD
	{
		SAMPLE_RATE = 30, // 30 Hz 
		MAX_CLIP_DURATION = 3, // 3 second for one cyclic clip
	};

	extern size_t CLIP_FRAME_COUNT; // default to 90 Frames for one clip

	template <typename Ty>
	struct KeyFrame // Meta data for an animation frame
	{
	public:
		std::string		Name;
		TimeScalarType	Time;
		Ty				Frame;
		//concept static Lerp(self&out, const self& lhs, const self& rhs, float t);

		//concept frame_type& operator[]
	};

	class LinearFrame
	{};
	class SpineFrame
	{};
	class DicredFrame
	{};

	class AnimationManager
	{

	};

	class LinearWarp
	{
	public:
		TimeScalarType operator()(TimeScalarType t)
		{
			return t;
		}
	};


	// The underline resources of an animation, not for "Play" Control
	// Handles Interpolations and warps in Time
	template <typename FrameType, typename FrameViewType>
	class KeyframeAnimation
	{
	public:
		typedef FrameType frame_type;
		typedef FrameViewType frame_view;


		string								Name;
		FrameType							DefaultFrame;
		std::list<KeyFrame<FrameType>>		KeyFrames;
		TimeScalarType						Duration;
		TimeScalarType						FrameInterval;
		bool								IsCyclic;
		// A function map : (BeginTime,EndTime) -> (BeginTime,EndTime),  that handels the easing effect between frames
		// Restriction : TimeWarp(KeyFrameTime(i)) must equals to it self
		/*std::function<TimeScalarType(TimeScalarType)> TimeWarpFunction;*/
	public:
		typedef KeyframeAnimation self_type;

		explicit KeyframeAnimation(const string& name)
			: Name(name)
		{}
		KeyframeAnimation() = default;
		KeyframeAnimation(const self_type& rhs) = default;
		//self_type& operator=(const self_type& rhs) = default;

		KeyframeAnimation(self_type&& rhs) = default;

		// Duration of this clip
		TimeScalarType	Length() const { return Duration; }
		// Is this clip a loopable animation : last frame == first frame
		bool			Cyclic()  const { return IsCyclic; }

		// Frame operations
	public:
		bool InsertKeyFrame(TimeScalarType time, const FrameType& frame);
		//bool InsertKeyFrame(TimeScalarType time, FrameType&& frame);
		bool InsertKeyFrame(int idx, const FrameType& frame);
		//bool InsertKeyFrame(int idx, FrameType&& frame);
		bool RemoveKeyFrame(int idx);
		const FrameType& GetKeyFrame(int idx) const;
		bool ReplaceKeyFrame(int idx, const FrameType& frame);
		bool Clear();

	public:
		// Frame Retrival
		virtual bool GetFrameAt(FrameViewType outFrame, TimeScalarType time) const;

	};

	template<typename FrameType, typename FrameViewType>
	inline bool KeyframeAnimation<FrameType, FrameViewType>::GetFrameAt(FrameViewType outFrame, TimeScalarType time) const
	{
		return false;
	}

	class ArmatureFrameBuffer
	{
	public:
		ArmatureFrameBuffer() : m_bones(0), m_frames(0) {}
		ArmatureFrameBuffer(size_t bones, size_t frames): m_bones(bones), m_frames(frames), m_buffer(bones*frames)
		{}
		ArmatureFrameBuffer(const IArmature& armature, size_t frames)
		: m_bones(armature.size()), m_frames(frames), m_buffer(armature.size()*frames)
		{}

		void resize(size_t joints, size_t frames)
		{
			m_buffer.resize(joints*frames);
		}
	
		void resize(size_t frames) {
			assert(m_bones > 0);
			resize(m_bones, frames);
		}

		size_t size() const { return m_frames; }
		size_t frames() const { return m_frames; }
		size_t bones() const { return m_bones; }

		void push_back(ArmatureFrameConstView frame)
		{
			assert(frame.size() == m_bones);
			m_buffer.insert(m_buffer.end(), frame.begin(), frame.end());
		}

		Bone& bone(size_t frame, size_t bid) { return m_buffer[frame*m_bones + bid]; }
		const Bone& bone(size_t frame, size_t bid) const { return m_buffer[frame*m_bones + bid]; }

		ArmatureFrameView at(size_t frame) { return ArmatureFrameView(&m_buffer[frame*m_bones], m_bones); }
		ArmatureFrameConstView at(size_t frame) const { return ArmatureFrameConstView(&m_buffer[frame*m_bones], m_bones); }

		inline ArmatureFrameView frame(size_t frameid) { return at(frameid); }
		inline ArmatureFrameConstView frame(size_t frameid) const { return at(frameid); }

		inline ArmatureFrameView operator[](size_t frame) { return at(frame); }
		inline ArmatureFrameConstView operator[](size_t frame) const { return at(frame); }

	private:
		size_t m_bones, m_frames;
		std::vector<Bone, DirectX::XMAllocator> m_buffer;
	};

	class ArmatureFrameAnimation : public KeyframeAnimation<ArmatureFrame, ArmatureFrameView>
	{
	public:
		typedef ArmatureFrameAnimation self_type;
		typedef KeyframeAnimation<ArmatureFrame, ArmatureFrameView> base_type;

		self_type() { pArmature = nullptr; }
		self_type(const self_type& rhs) = default;
		//self_type& operator=(const self_type& rhs) = default;

		self_type(self_type&& rhs) = default;
		//{
		//	*this = std::move(rhs);
		//}

		explicit self_type(const std::string& name);
		explicit self_type(std::istream& file);

		//self_type& operator=(const self_type&& rhs);

		const IArmature& Armature() const { return *pArmature; }
		void SetArmature(const IArmature& armature);
		// get the pre-computed frame buffer which contains interpolated frame
		const std::vector<frame_type>& GetFrameBuffer() const { return frames; }
		std::vector<frame_type>& GetFrameBuffer() { return frames; }

		bool InterpolateFrames(double frameRate);
		bool GetFrameAt(frame_view outFrame, TimeScalarType time) const override;

		enum DataType
		{
			LocalRotation = 0,
			GlobalRotation = 2,
			OriginPosition = 3,
			EndPositon = 4,
		};

		// Eigen interface
		//Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::Stride<sizeof(float), sizeof(Bone)>> DataMatrix(DataType data) const
		//{
		//	return Eigen::Map<Eigen::Matrix3Xf, Eigen::Aligned, Eigen::Stride<sizeof(float), sizeof(Bone)>>(&frames[0][0]);
		//}

		void Serialize(std::ostream& binary) const;
		void Deserialize(std::istream& binary);

	private:
		const IArmature*			pArmature;
		size_t						bonesCount;
		std::vector<frame_type>		frames;
	//public:
	//	Eigen::MatrixXf		animMatrix; // 20N x F matrix
	//	Eigen::RowVectorXf  Ecj;	// Jointwise Energy
	//	Eigen::RowVectorXf	Ecb;	// Blockwise Energy
	//	Eigen::Matrix<float,3,-1> Ysp;	// Spatial traits
	//	std::vector<Eigen::QrStore<Eigen::MatrixXf>> QrYs;
	//	std::vector<Eigen::Pca<Eigen::MatrixXf>> PcaYs;
	//	std::vector<Eigen::MatrixXf> Ys;
	};

	class ArmatureTransform
	{
	public:

		ArmatureTransform();
		virtual ~ArmatureTransform()
		{
		}

		typedef ArmatureFrame frame_type;
		typedef ArmatureFrameView frame_view;
		typedef ArmatureFrameConstView const_frame_view;

		const IArmature& SourceArmature() const { return *m_sArmature; }
		const IArmature& TargetArmature() const { return *m_tArmature; }
		void SetSourceArmature(const IArmature& armature) { m_sArmature = &armature; }
		void SetTargetArmature(const IArmature& armature) { m_tArmature = &armature; }

		int GetBindIndex(int sourceIdx) const;
		int GetInverseBindIndex(int tragetIdx) const;

		virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame) = 0;

		// Transform with history data
		virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, float frame_time);

		virtual void TransformBack(frame_view target_frame, const_frame_view source_frame);

	protected:
		const IArmature	*m_sArmature, *m_tArmature;
	};

	template <typename _FrameType>
	class IStreamAnimation abstract
	{
	public:
		typedef _FrameType FrameType;
		// Advance the stream by 1 
		virtual bool ReadNextFrame() = 0;

		// Advance the stream to latest
		virtual bool ReadLatestFrame()
		{
			bool flag = false;
			while (ReadNextFrame())
				flag = true;
			return flag;
		}

		// Peek the current frame head
		virtual const FrameType& PeekFrame() const = 0;

		virtual bool IsAvailable() const = 0;
	};

	class IArmatureStreamAnimation : public IStreamAnimation<ArmatureFrame>
	{
	public:
		typedef ArmatureFrame frame_type;

		virtual const IArmature& GetArmature() const = 0;
	};

	enum AnimationPlayState
	{
		Stopped = 0,
		Active = 1,
		Paused = 2,
	};

	enum AnimationRepeatBehavior
	{
		NotRepeat = 0,
		AlwaysRepeat = 1,
	};


// Stupid windows header
#if defined(CetCurrentTime)
#pragma push_macro("GetCurrentTime")
#undef CetCurrentTime
#define PUSHED_GET_CURRENT_TIME 1
#endif

	class StoryBoard
	{
	public:
		// Play Control
		void Begin();
		void Stop();
		void Pause();
		void Resume();

		// Play properties
		AnimationRepeatBehavior GetRepeatBehavior() const;
		void SetRepeatBehavior(AnimationRepeatBehavior behavior);

		double GetSpeedRatio() const;
		void SetSpeedRatio(double ratio) const;

		// Timeline basic
		TimeScalarType GetBeginTime() const;
		void SetBeginTime(TimeScalarType time);
		TimeScalarType GetEndTime() const;
		void SetEndTime(TimeScalarType time);
		TimeScalarType GetDuration();

		// Time seeking
		AnimationPlayState GetCurrentState() const;
		TimeScalarType GetCurrentTime() const; // Shit! Name Collision with Windows Header!!! Dame te outdated C-style header!!!
		void AdvanceTimeBy(TimeScalarType time);
		void Seek(TimeScalarType time);

		//TypedEvent<StoryBoard> Completed;

		//std::vector<IFrameAnimation*> ChildrenAnimation;
	protected:
		AnimationPlayState CurrentState;
	};

#if defined(PUSHED_GET_CURRENT_TIME)
#pragma pop_macro("GetCurrentTime")
#undef PUSHED_GET_CURRENT_TIME
#endif

}