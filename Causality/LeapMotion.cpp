#include "pch_bcl.h"
#include "LeapMotion.h"
#include <iostream>
#include <Leap.h>

//#ifdef _DEBUG
//#pragma comment(lib,"Leapd.lib")
//#else
#pragma comment(lib,"Leap.lib")
//#endif

using namespace Causality;
using namespace Causality::Devices;
using namespace Causality::Devices::Internal;

unique_ptr<StaticArmature> TrackedHand::s_pHandArmature;

#pragma region HandArmatureStaticDefination
static const char * const HandJointNames[TrackedHand::JointType_Count]=
{
	"Wrist",
	"ThumbMetacarpal",
	"ThumbProximal",
	"ThumbIntermediate",
	"ThumbDistal",
	"IndexMetacarpal",
	"IndexProximal",
	"IndexIntermediate",
	"IndexDistal",
	"MiddleMetacarpal",
	"MiddleProximal",
	"MiddleIntermediate",
	"MiddleDistal",
	"RingMetacarpal",
	"RingProximal",
	"RingIntermediate",
	"RingDistal",
	"PinkyMetacarpal",
	"PinkyProximal",
	"PinkyIntermediate",
	"PinkyDistal"
};

int TrackedHand::s_HandArmatureParentMap[TrackedHand::JointType_Count]=
{
	JointType_Null,
	JointType_Wrist,
	JointType_ThumbMetacarpal,
	JointType_ThumbProximal,
	JointType_ThumbIntermediate,
	JointType_Wrist,
	JointType_IndexMetacarpal,
	JointType_IndexProximal,
	JointType_IndexIntermediate,
	JointType_Wrist,
	JointType_MiddleMetacarpal,
	JointType_MiddleProximal,
	JointType_MiddleIntermediate,
	JointType_Wrist,
	JointType_RingMetacarpal,
	JointType_RingProximal,
	JointType_RingIntermediate,
	JointType_Wrist,
	JointType_PinkyMetacarpal,
	JointType_PinkyProximal,
	JointType_PinkyIntermediate,
};
#pragma endregion

void TrackedHand::IntializeHandArmature()
{
	if (s_pHandArmature) return;
	s_pHandArmature.reset(new StaticArmature((size_t)JointType_Count, s_HandArmatureParentMap, HandJointNames));
}

class Devices::Internal::LeapListener : public Leap::Listener
{
public:
	LeapListener(LeapSensor* pLeap)
	{
		m_pSensor = pLeap;
		m_pause = false;
		m_hasNewFrame = false;
		m_lostThreshold = 30;
	}
	~LeapListener()
	{
	}

	LeapListener(const LeapListener&) = delete;

	virtual void onConnect(const Leap::Controller & controller) override;
	virtual void onDisconnect(const Leap::Controller &controller) override;
	virtual void onFrame(const Leap::Controller &controller) override;

	void processLostHands();

	bool pullFrame(const Leap::Controller & controller);

	void processHand(const Leap::Hand& leaphand, TrackedHand& hand, bool isNew, time_t time);

	LeapSensor*					m_pSensor;
	Leap::Frame					m_frame;
	Matrix4x4					m_world;
	int							m_lostThreshold;
	bool						m_pause;
	bool						m_hasNewFrame;
	std::list<TrackedHand>		m_hands;
};

std::weak_ptr<LeapSensor> LeapSensor::wpCurrentDevice;

std::shared_ptr<LeapSensor> Causality::Devices::LeapSensor::GetForCurrentView()
{
	if (wpCurrentDevice.expired())
	{
		auto pDevice = std::make_shared<LeapSensor>(false,true);
		wpCurrentDevice = pDevice;
		return pDevice;
	}
	else
	{
		return wpCurrentDevice.lock();
	}
}

LeapSensor::LeapSensor()
	:
	m_useEvent(false),
	m_isFixed(true),
	m_started(false),
	m_connected(false)
{
	pController.reset(new Leap::Controller());
	pListener.reset(new LeapListener(this));
}

LeapSensor::LeapSensor(bool useEvent, bool isFixed)
	: LeapSensor()
{
	Initialize(useEvent, isFixed);
}

void LeapSensor::Initialize(bool useEvent, bool isFixed)
{
	TrackedHand::IntializeHandArmature();

	if (m_started)
		Stop();

	using namespace DirectX;
	m_useEvent = useEvent;
	m_isFixed = isFixed;
	if (!isFixed)
		pController->setPolicy(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
}

LeapSensor::~LeapSensor()
{
	if (pListener && pController)
		pController->removeListener(*pListener.get());
	pListener.reset();
	pController.reset();
}

Leap::Controller & LeapSensor::Controller() {
	return *pController;
}

const Leap::Controller & LeapSensor::Controller() const {
	return *pController;
}

//void LeapSensor::SetMotionProvider(DirectX::ILocatable * pHeadLoc, DirectX::IOriented * pHeadOrient)
//{
//	pHeadLocation = pHeadLoc;
//	pHeadOrientation = pHeadOrient;
//}

void LeapSensor::SetDeviceWorldCoord(const DirectX::Matrix4x4 & m)
{
	using namespace DirectX;
	if (pListener)
		pListener->m_world = m;
}

DirectX::XMMATRIX LeapSensor::ToWorldTransform() const
{
	using namespace DirectX;
	//if (pHeadLocation && pHeadOrientation)
	//	return m_world * XMMatrixRigidTransform(pHeadOrientation->GetOrientation() ,pHeadLocation->GetPosition());
	//else
	return pListener->m_world;
}

void LeapListener::onConnect(const Leap::Controller & controller)
{
	m_pSensor->Connected(*m_pSensor);
	std::cout << "[Leap] Device Connected." << std::endl;
}

void LeapListener::onDisconnect(const Leap::Controller & controller)
{
	m_pSensor->Disconnected(*m_pSensor);
	std::cout << "[Leap] Device Disconnected." << std::endl;
}

void LeapListener::onFrame(const Leap::Controller & controller)
{
	if (m_pause)
		m_hasNewFrame = false;
	else
		m_hasNewFrame = pullFrame(controller);	//m_pSensor->FrameArrived(controller);

	if (!m_hasNewFrame)	return;

	// Process experied bodies
	processLostHands();

	for (const auto& leaphand : m_frame.hands())
	{
		auto itr = std::find(m_hands.begin(), m_hands.end(), (int64_t)leaphand.id());
		bool isNew = itr == m_hands.end();
		if (isNew && leaphand.isValid())
		{
			itr = m_hands.emplace(m_hands.end(), (int64_t)leaphand.id(), leaphand.isLeft());
			itr->m_pSensor = m_pSensor;
		}
		else if (!isNew && !leaphand.isValid()) // this means not tracked
		{
			continue;
		}

		auto& hand = *itr;

		processHand(leaphand, hand, isNew, m_frame.timestamp());

		if (isNew)
		{
			auto& hand = m_hands.back();
			m_pSensor->HandTracked(hand);
		}
	}
}

void LeapListener::processLostHands()
{
	for (auto itr = m_hands.begin(), end = m_hands.end(); itr != end;)
	{
		itr->IncreaseLostFrameCount();
		if (itr->GetLostFrameCount() >= m_lostThreshold)
		{
			if (itr->m_isTracked)
			{
				itr->m_isTracked = false;
				itr->Lost(*itr);
				m_pSensor->HandLost(*itr);
				int refcount = itr->RefCount();
				std::cout << "Hands Lost : ID = " << itr->GetTrackId() << "RefCount = " << refcount << std::endl;
			}
			if (itr->RefCount() <= 0)
				itr = m_hands.erase(itr);
			else
				++itr;
		}
		else
			++itr;
	}

}

bool LeapListener::pullFrame(const Leap::Controller & controller)
{
	auto frame = controller.frame();
	if (!frame.isValid() || m_frame == frame)
		return false;
	m_frame = std::move(frame);
	return true;
}


void LeapListener::processHand(const Leap::Hand& leaphand, TrackedHand& hand, bool isNew, time_t time)
{
	Vector3 s, t;
	ArmatureFrame frame(TrackedHand::JointType_Count);

	hand.m_isTracked = true;
	hand.ResetLostFrameCount();

	bool isLeft = leaphand.isLeft();

	auto lbasis = leaphand.basis();
	lbasis.origin = leaphand.wristPosition();
	if (isLeft)	lbasis.zBasis *= -1;

	auto basis = lbasis.toMatrix4x4<Matrix4x4>();
	basis *= m_world;
	basis.Decompose(s, frame[0].GblRotation, frame[0].GblTranslation);

	for (auto& finger : leaphand.fingers())
	{
		auto fid = finger.type();
		for (int bid = Leap::Bone::Type::TYPE_METACARPAL;
				 bid <= Leap::Bone::TYPE_DISTAL; ++bid)
		{
			auto leapbone = finger.bone((Leap::Bone::Type)bid);
			auto& bone = frame[fid*4 + bid+1];
			auto lbasis = leapbone.basis();
			lbasis.origin = leapbone.nextJoint();
			if (isLeft)
				lbasis.zBasis *= -1;
			basis = lbasis.toMatrix4x4<Matrix4x4>();
			basis *= m_world;
			basis.Decompose(s, bone.GblRotation, bone.GblTranslation);
		}
	}

	auto& armature = hand.m_armature;
	FrameRebuildLocal(armature, frame);

	if (isNew)
		hand.SetArmatureProportion(frame);

	hand.PushFrame(time,std::move(frame));

	if (isNew)
	{
		m_pSensor->HandTracked(hand);
		std::cout << "New Hand Tracked : ID = " << hand.GetTrackId() << std::endl;
	}
}

bool LeapSensor::Initialize(const ParamArchive * archive)
{
	IsometricTransform world;
	world.Scale = Vector3(0.001f); // convert unit mm to m , factor 1mm = 0.01m
	GetParam(archive, "translation", world.Translation);
	GetParam(archive, "scale", world.Scale);
	Vector3 eular;
	GetParam(archive, "rotation", eular);
	world.Rotation = Quaternion::CreateFromYawPitchRoll(eular.y, eular.x, eular.z);
	SetDeviceWorldCoord(world.TransformMatrix());

	bool useEvent = false, isFixed = true;
	GetParam(archive, "asynchronize", useEvent);
	GetParam(archive, "fixed", isFixed);
	Initialize(useEvent, isFixed);

	return true;
}

bool LeapSensor::Start()
{
	if (m_started) return false;

	if (m_useEvent)
		pController->addListener(*pListener);
	m_started = true;
	return m_started;
}

void LeapSensor::Stop()
{
	if (!Disconnected.empty() && m_connected)
	{
		m_connected = false;
		Disconnected(*this);
	}

	if (m_started && m_useEvent)
		pController->removeListener(*pListener);
}

bool LeapSensor::Update()
{
	bool connected = pController->isConnected();
	bool prev = m_connected;

	if (prev != connected)
		m_connected = connected;

	if (connected && !prev)
		pListener->onConnect(*pController);
	else if (!connected && prev)
		pListener->onDisconnect(*pController);

	if (connected)
	{
		pListener->onFrame(*pController);
		return pListener->m_hasNewFrame;
	}
	return connected;
}

bool LeapSensor::IsStreaming() const
{
	return !pListener && pController->isConnected() && m_started;
}

bool LeapSensor::IsAsychronize() const
{
	return m_useEvent;
}

bool LeapSensor::Pause()
{
	pListener->m_pause = true;
	return true;
}

bool LeapSensor::Resume()
{
	pListener->m_pause = false;
	return true;
}

const std::list<TrackedHand>& Causality::Devices::LeapSensor::GetTrackedHands() const
{
	return pListener->m_hands;
}

std::list<TrackedHand>& Causality::Devices::LeapSensor::GetTrackedHands()
{
	return pListener->m_hands;
}

TrackedHand::TrackedHand(int64_t id, bool isLeft)
	: TrackedArmature(*s_pHandArmature)
{
	m_id = id;
	m_isLeft = isLeft;
}

TrackedHand::~TrackedHand()
{
}
