#include "pch_bcl.h"
#include "Animations.h"

using namespace Causality;
using namespace DirectX;
using namespace Eigen;
using namespace std;

size_t Causality::CLIP_FRAME_COUNT = 90U;

ArmatureFrameAnimation::ArmatureFrameAnimation(std::istream & file)
{
	using namespace std;
	using namespace DirectX;
	self_type animation;
	int keyframs, joints;
	file >> joints >> keyframs;
	animation.KeyFrames.resize(keyframs);
	auto itrKey = KeyFrames.begin();
	for (int i = 0; i < keyframs; i++)
	{
		auto& key = (*itrKey++);
		key.Frame.resize(joints);
		double seconds;
		file >> key.Name;
		file >> seconds;
		key.Time = (time_seconds)seconds;
		for (auto& bone : key.Frame)
		{
			Vector3 vec;
			auto& scl = bone.LclScaling;
			//char ch;
			file >> vec.x >> vec.y >> vec.z >> scl.y;
			bone.LclRotation = XMQuaternionRotationRollPitchYawFromVector(vec);
		}
	}
}

ArmatureFrameAnimation::ArmatureFrameAnimation(const std::string & name)
	: base_type(name)
{
}

bool ArmatureFrameAnimation::InterpolateFrames(double frameRate)
{
	float delta = (float)(1.0 / frameRate);
	auto& armature = *pArmature;
	float t = 0;

	auto itrKey = KeyFrames.begin();
	auto itrKeyAdv = ++KeyFrames.begin();

	for (size_t i = 1; i < KeyFrames.size(); i++)
	{
		const frame_type& lhs = itrKey->Frame;
		const frame_type& rhs = itrKeyAdv->Frame;
		for (t = (float)itrKey->Time.count(); t < (float)itrKey->Time.count(); t += delta)
		{
			frames.emplace_back(armature.size());
			FrameLerp(frames.back(), lhs, rhs, (float)t, armature);
		}
		++itrKey;
		++itrKeyAdv;
	}
	frames.shrink_to_fit();
	return true;
}

bool ArmatureFrameAnimation::GetFrameAt(frame_view outFrame, TimeScalarType time) const
{

	double t = fmod(time.count(), Duration.count());
	if (t < 0) t += Duration.count();
	int frameIdx = (int)floorf(t / FrameInterval.count());
	frameIdx = frameIdx % frames.size(); // ensure the index is none negative
	auto& sframe = frames[frameIdx];
	auto& tframe = frames[(frameIdx + 1) % frames.size()];
	t -= frameIdx * FrameInterval.count();
	t /= FrameInterval.count();

	FrameLerp(outFrame, sframe, tframe,t, Armature());
	return true;
}

void ArmatureFrameAnimation::Serialize(std::ostream & binary) const
{
	binary << (uint32_t)bonesCount << (uint32_t)frames.size() << (uint32_t)sizeof(Bone);
	binary << (double)Duration.count() << (double)FrameInterval.count();
	for (auto& frame : frames)
	{
		binary.write(reinterpret_cast<const char*>(frame.data()),
			sizeof(Bone)*bonesCount);
	}
}

void ArmatureFrameAnimation::Deserialize(std::istream & binary)
{
	uint32_t bC, fC, sB;
	double dur, itv;
	binary >> bC >> fC >> sB >> dur >> itv;

	Duration = time_seconds(dur);
	FrameInterval = time_seconds(itv);

	assert((int)(dur / itv) == fC);
	assert(sB == sizeof(Bone));

	time_seconds time(0);
	frames.resize(fC);
	for (auto& frame : frames)
	{
		time += FrameInterval;

		frame.resize(bC);
		binary.read(reinterpret_cast<char*>(frame.data()),
			sizeof(Bone)*bonesCount);
	}
}

ArmatureTransform::ArmatureTransform() {
	m_sArmature = nullptr;
	m_tArmature = nullptr;
}

// Transform with history data

void Causality::ArmatureTransform::Transform(frame_view target_frame, const_frame_view source_frame, const_frame_view last_frame, float frame_time)
{
	Transform(target_frame, source_frame);
}

void ArmatureTransform::TransformBack(frame_view target_frame, const_frame_view source_frame)
{
}