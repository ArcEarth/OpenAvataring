#include "pch.h"
#include "PlayerSelector.h"
#include <Causality\Kinect.h>

using namespace Causality;
using namespace Causality::Devices;

KinectPlayerSelector::KinectPlayerSelector(KinectSensor * pKinect, SelectionMode mode)
	:pCurrent(nullptr), pKinect(nullptr), mode(None)
{
	Initialize(pKinect, mode);
}

KinectPlayerSelector::~KinectPlayerSelector()
{
	fpTrackedBodyChanged = nullptr;
	fpFrameArrived = nullptr;

	ChangePlayer(nullptr);

	if (pKinect)
	{
		con_tracked.disconnect();
		con_lost.disconnect();
	}
}

void KinectPlayerSelector::OnPlayerTracked(TrackedBody & body)
{
	if ((!pCurrent || !(mode & Sticky)))
	{
		if (!pCurrent || (mode == SelectionMode::Closest && body.DistanceToSensor() < pCurrent->DistanceToSensor()))
		{
			ChangePlayer(&body);
		}
	}
}

void KinectPlayerSelector::OnPlayerLost(TrackedBody & body)
{
	if (pCurrent && body == *pCurrent)
	{
		ReSelectFromAllTrackedBodies();
	}
}

void KinectPlayerSelector::ReSelectFromAllTrackedBodies()
{
	TrackedBody *pBestPlayer = nullptr;

	if (mode & Closest)
	{
		float distance = 100000;
		for (auto& player : pKinect->GetTrackedBodies())
		{
			if (player.IsTracked() &&(pCurrent ==nullptr || player != *pCurrent && distance > player.DistanceToSensor()))
			{
				pBestPlayer = &player;
				distance = player.DistanceToSensor();
			}
		}
	}
	else // Eariest tracked player
	{
		for (auto& player : pKinect->GetTrackedBodies())
		{
			if (player.IsTracked() && (pCurrent == nullptr || player != *pCurrent))
			{
				pBestPlayer = &player;
				break;
			}
		}
	}

	ChangePlayer(pBestPlayer);

}

void KinectPlayerSelector::Reset() {

	if (fpTrackedBodyChanged)
		fpTrackedBodyChanged(pCurrent, nullptr);
	if (pCurrent)
		pCurrent->Release();
	pCurrent = nullptr;
}

void KinectPlayerSelector::Initialize(Devices::KinectSensor * pKinect, SelectionMode mode)
{
	this->mode = mode;
	if (pKinect)
	{
		this->pKinect = pKinect->GetRef();
		con_tracked =
			pKinect->OnPlayerTracked += MakeEventHandler(&KinectPlayerSelector::OnPlayerTracked, this);
		con_lost =
			pKinect->OnPlayerLost += MakeEventHandler(&KinectPlayerSelector::OnPlayerLost, this);

		if (pKinect->GetTrackedBodies().size() > 0)
		{
			ReSelectFromAllTrackedBodies();
		}
	}
}

void KinectPlayerSelector::ChangePlayer(TrackedBody * pNewPlayer)
{
	auto pOld = pCurrent;
	con_frame.disconnect();
	pCurrent = pNewPlayer;

	if (pCurrent)
	{
		pCurrent->AddRef();
		if (fpFrameArrived)
			con_frame = pCurrent->OnFrameArrived.connect(fpFrameArrived);
	}

	if (fpTrackedBodyChanged)
		fpTrackedBodyChanged(pOld, pCurrent);

	if (pOld)
		pOld->Release();
}

void KinectPlayerSelector::SetFrameCallback(const FrameEventFunctionType & callback) {
	fpFrameArrived = callback;

	if (con_frame.connected())
		con_frame.disconnect();

	if (fpFrameArrived && pCurrent)
		con_frame = pCurrent->OnFrameArrived.connect(fpFrameArrived);
}

void KinectPlayerSelector::SetPlayerChangeCallback(const PlayerEventFunctionType & callback) {
	fpTrackedBodyChanged = callback;
	if (pCurrent && fpTrackedBodyChanged)
	{
		fpTrackedBodyChanged(nullptr, pCurrent);
	}
}

IArmatureStreamAnimation * KinectPlayerSelector::Get() { return pCurrent; }

void KinectPlayerSelector::ChangeSelectionMode(SelectionMode mdoe)
{
	this->mode = mode;
	if (pCurrent)
		ReSelectFromAllTrackedBodies();
}

