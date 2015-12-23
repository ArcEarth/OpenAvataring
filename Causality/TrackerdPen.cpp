#include "pch_bcl.h"
#include "TrackerdPen.h"
#include "BasicKeyboardMouseControlLogic.h"
#include "LeapMotion.h"
#include "Vicon.h"
#include "Scene.h"
#include "CameraObject.h"

using namespace Causality;
using namespace Math;

REGISTER_SCENE_OBJECT_IN_PARSER(tracked_pen, TrackedPen);

const float TrackedPen::TipLength = 0.0145f;

TrackedPen::TrackedPen()
{
	m_visible = false;
	m_inkingStr = 0;
	m_dragingStr = 0;
	m_erasingStr = 0;
}

TrackedPen::~TrackedPen()
{

}

void TrackedPen::OnParentChanged(SceneObject * oldParent)
{
	TrackedObjectControl::OnParentChanged(oldParent);
	auto pCamera = dynamic_cast<SceneObject*>(this->Scene->PrimaryCamera());
	if (pCamera)
		setMouse(pCamera->FirstChildOfType<KeyboardMouseFirstPersonControl>());
}

void TrackedPen::Update(const time_seconds & time_delta)
{
	SceneObject::Update(time_delta);
	if (m_pVicon)
		m_visible = UpdateFromVicon(time_delta.count());
	else if (m_pLeap)
		m_visible = UpdateFromLeapFinger(time_delta.count());
}

bool TrackedPen::UpdateFromLeapFinger(double dt)
{
	auto object = m_pRigid;
	if (!object) return false;
	auto frame = m_pLeap->Controller().frame();
	XMMATRIX world = m_pLeap->ToWorldTransform();

	auto& hands = frame.hands();
	for (auto& hand : hands)
	{
		if (m_idx && hand.isRight() || !m_idx && hand.isLeft())
		{
			m_inkingStr = hand.pinchStrength();
			m_dragingStr = hand.grabStrength();

			auto indexFingerList = hand.fingers().fingerType(Leap::Finger::Type::TYPE_INDEX);
			auto indexFinger = *indexFingerList.begin(); //since there is only one per hand

			XMVECTOR pos = indexFinger.tipPosition().toVector3<Vector3>();
			//XMVECTOR pos = hand.palmPosition().toVector3<Vector3>();

			pos = XMVector3TransformCoord(pos, world);

			XMVECTOR xdir = hand.palmNormal().toVector3<Vector3>();
			XMVECTOR ydir = indexFinger.bone(Leap::Bone::Type::TYPE_DISTAL).direction().toVector3<Vector3>();
			//XMVECTOR ydir = hand.direction().toVector3<Vector3>();
			XMVECTOR zdir = XMVector3Cross(xdir, ydir);

			XMMATRIX rot = XMMatrixIdentity();
			rot.r[0] = xdir;
			rot.r[1] = ydir;
			rot.r[2] = zdir;

			//! HACK, since rot(world) == I
			XMVECTOR rotQ = XMQuaternionRotationMatrix(rot);

			object->SetPosition(pos);
			object->SetOrientation(rotQ);
			return true;
		}
	}
	return false;
}

bool TrackedPen::IsInking() const
{
	return (m_pVicon != nullptr && m_mouse->isLeftButtonDown())
		|| (m_pLeap != nullptr && (m_inkingStr > 0.5f && m_inkingStr > m_dragingStr && m_inkingStr > m_erasingStr));
	// Was for leap:
	//return ;
}

bool TrackedPen::IsErasing() const
{
	return m_pLeap != nullptr && m_erasingStr > 0.5f && m_erasingStr > m_dragingStr && m_erasingStr > m_inkingStr;
}

bool TrackedPen::IsDraging() const
{
	return (m_pVicon != nullptr && m_mouse->isRightButtonDown())
		|| (m_pLeap != nullptr && (m_dragingStr > 0.5f && m_dragingStr > m_inkingStr && m_dragingStr > m_erasingStr));
}

bool TrackedPen::IsVisible() const
{
	return m_visible;
}

void TrackedPen::setMouse(const KeyboardMouseFirstPersonControl * m) { m_mouse = m; }
