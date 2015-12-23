#include "pch_bcl.h"
#include "LeapMotion.h"
#include <iostream>

//#ifdef _DEBUG
//#pragma comment(lib,"Leapd.lib")
//#else
#pragma comment(lib,"Leap.lib")
//#endif

using namespace Causality::Devices;

class LeapMotion::Listener : public Leap::Listener
{
public:
	Listener(LeapMotion* pLeap)
	{
		pOwner = pLeap;
		PrevHandsCount = 0;
	}
	virtual void onConnect(const Leap::Controller & controller) override;
	virtual void onDisconnect(const Leap::Controller &controller) override;
	virtual void onFrame(const Leap::Controller &controller) override;

	int							PrevHandsCount;
	Leap::Frame					CurrentFrame;
	LeapMotion*	pOwner;
};

std::weak_ptr<LeapMotion> LeapMotion::wpCurrentDevice;

std::shared_ptr<LeapMotion> Causality::Devices::LeapMotion::GetForCurrentView()
{
	if (wpCurrentDevice.expired())
	{
		auto pDevice = std::make_shared<LeapMotion>(false,true);
		wpCurrentDevice = pDevice;
		return pDevice;
	}
	else
	{
		return wpCurrentDevice.lock();
	}
}

LeapMotion::LeapMotion(bool useEvent, bool isFixed)
	:pListener(new Listener(this))
{
	using namespace DirectX;
	PrevConnectionStates = false;
	//pHeadLocation = nullptr;
	//pHeadOrientation = nullptr;
	Coordinate = XMMatrixScalingFromVector(XMVectorReplicate(0.001f));// Default setting
	if (useEvent)
		LeapController.addListener(*pListener.get());
	if (!isFixed)
		LeapController.setPolicy(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
}

LeapMotion::~LeapMotion()
{
	LeapController.removeListener(*pListener.get());
}

Leap::Controller & LeapMotion::Controller() {
	return LeapController;
}

const Leap::Controller & LeapMotion::Controller() const {
	return LeapController;
}

void LeapMotion::PullFrame()
{
	bool connected = LeapController.isConnected();
	if (connected && !PrevConnectionStates)
	{
		PrevConnectionStates = connected;
		pListener->onConnect(LeapController);
	}
	else if (!connected && PrevConnectionStates)
	{
		PrevConnectionStates = connected;
		pListener->onDisconnect(LeapController);
	}
	if (connected)
		pListener->onFrame(LeapController);
}

//void LeapMotion::SetMotionProvider(DirectX::ILocatable * pHeadLoc, DirectX::IOriented * pHeadOrient)
//{
//	pHeadLocation = pHeadLoc;
//	pHeadOrientation = pHeadOrient;
//}

void LeapMotion::SetDeviceWorldCoord(const DirectX::Matrix4x4 & m)
{
	using namespace DirectX;
	Coordinate = XMMatrixScalingFromVector(XMVectorReplicate(0.001f)) * (XMMATRIX)m;
}

DirectX::XMMATRIX LeapMotion::ToWorldTransform() const
{
	using namespace DirectX;
	//if (pHeadLocation && pHeadOrientation)
	//	return Coordinate * XMMatrixRigidTransform(pHeadOrientation->GetOrientation() ,pHeadLocation->GetPosition());
	//else
	return Coordinate;
}

void LeapMotion::Listener::onConnect(const Leap::Controller & controller)
{
	pOwner->DeviceConnected(controller);
	PrevHandsCount = 0;
	std::cout << "[Leap] Device Connected." << std::endl;
}

void LeapMotion::Listener::onDisconnect(const Leap::Controller & controller)
{
	pOwner->DeviceConnected(controller);
	PrevHandsCount = 0;
	std::cout << "[Leap] Device Disconnected." << std::endl;
}

void LeapMotion::Listener::onFrame(const Leap::Controller & controller)
{
	CurrentFrame = controller.frame();
	pOwner->FrameArrived(controller);

	UserHandsEventArgs e{ controller , pOwner->ToWorldTransform()};

	if (CurrentFrame.hands().count() > PrevHandsCount)
	{
		PrevHandsCount = CurrentFrame.hands().count();
		pOwner->HandsTracked(e);
	}
	else if (CurrentFrame.hands().count() < PrevHandsCount)
	{
		PrevHandsCount = CurrentFrame.hands().count();
		pOwner->HandsLost(e);
		//pOwner->HandsMove(e);
	}
	if (CurrentFrame.hands().count() > 0)
	{
		pOwner->HandsMove(e);
	}

	//std::cout << "[Leap] Frame arrived." << std::endl;
}
