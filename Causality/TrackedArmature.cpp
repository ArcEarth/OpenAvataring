#include "pch_bcl.h"
#include "TrackedArmature.h"

using namespace std;
using namespace Causality;

typedef TrackedArmature::FrameType FrameType;

TrackedArmature::TrackedArmature(size_t bufferSize)
	:m_frameBuffer(bufferSize)
{
}

TrackedArmature::TrackedArmature(const IArmature& armature, size_t bufferSize)
	:m_frameBuffer(bufferSize), m_armature(armature)
{
}

void TrackedArmature::PushFrame(time_t time, FrameType && frame)
{
    m_lastTrackedTime = time;
	m_frameBuffer.Push(std::move(frame));
	FireFrameArrivedForLatest();
}

bool TrackedArmature::ReadLatestFrame()
{
	return m_frameBuffer.MoveToLatest();
}

bool TrackedArmature::ReadNextFrame()
{
	return m_frameBuffer.MoveNext();
}

const FrameType& TrackedArmature::PeekFrame() const
{
	return *m_frameBuffer.GetCurrent();
}

bool TrackedArmature::IsAvailable() const
{
	return m_isTracked;
}

void TrackedArmature::SetArmature(const IArmature & armature)
{
	m_armature.clone_from(armature);
}

const IArmature & TrackedArmature::GetArmature() const {
	return m_armature;
}

void replaceLclTranslation(_Out_ ArmatureFrame& frame, _In_ ArmatureFrameConstView reframe)
{
	using namespace DirectX;
	for (int i = 0; i < frame.size(); i++)
	{
		auto& dbone = frame[i];
		auto& rbone = reframe[i];
		XMVECTOR V = XMVector3InverseRotate(
			XMLoadA(rbone.LclTranslation),
			XMLoadA(rbone.LclRotation));
		V = XMVector3Rotate(V, dbone.LclRotation);
		dbone.LclLength = XMVectorGetX(XMVector3Length(V));
		XMStoreA(dbone.LclTranslation , V);
	}
}

void TrackedArmature::SetArmatureProportion(ArmatureFrameConstView frameView)
{
	auto& dframe = m_armature.bind_frame();
	replaceLclTranslation(dframe, frameView);
	FrameRebuildGlobal(m_armature, dframe);
}

void TrackedArmature::FireFrameArrivedForLatest()
{
	m_frameBuffer.LockBuffer();
	auto pFrame = m_frameBuffer.PeekLatest();
	if (!pFrame)
	{
		m_frameBuffer.UnlockBuffer();
		return;
	}

	assert(pFrame != nullptr); // since we just pushed one frame 
	if (!OnFrameArrived.empty())
		OnFrameArrived(*this, *pFrame);
	m_frameBuffer.UnlockBuffer();
}

TrackedObject::TrackedObject()
: m_refCount(0), m_id(0), m_isTracked(false), m_lastTrackedTime(0), m_lostFrameCount(0)
{
}

void TrackedObject::AddRef() {
	++m_refCount;
}

void TrackedObject::Release() {
	--m_refCount;
}