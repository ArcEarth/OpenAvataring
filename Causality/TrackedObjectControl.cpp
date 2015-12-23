#include "pch_bcl.h"
#include "TrackedObjectControl.h"
#include "LeapMotion.h"

#include "Vicon.h"


using namespace Causality;
using namespace Causality::Devices;
using namespace Math;

REGISTER_SCENE_OBJECT_IN_PARSER(tracked_object, TrackedObjectControl);

bool TrackedObjectControl::UpdateFromVicon(double dt)
{
	m_freq = 1.0 / dt;

	if (m_pVicon)
	{
		auto object = m_pRigid;
		// name of the rigid body
		string viconName = object->Name + ':' + object->Name;
		// name of some marker in there will be objName:Mark1
		auto id = m_pVicon->GetRigidID(viconName);
		if (id == -1) return false;

		auto rigid = m_pVicon->GetRigid(id);

		RigidTransform calibrate;
		RigidTransform temp2;
		//calibrate.Translation.x = -0.0145;
		//calibrate.Translation.y = 0.0075f;
		calibrate = rigid.Inversed();
		//temp2 *= rigid;

		Vector4 cq = calibrate.Rotation;
		//std::cout << object->Name << " : translation=\"" << calibrate.Translation << "\" rotation=\"{" << cq << "}\"" << std::endl;

		RigidTransform fin = m_intrinsic;
		fin *= rigid;

		if (Vector3::Distance(rigid.Translation,Vector3::Zero) < 0.001f)
			return true;

		fin.Translation = m_posFilter.Apply(fin.Translation);
		fin.Rotation = m_rotFilter.Apply(fin.Rotation);

		object->SetPosition(fin.Translation);
		object->SetOrientation(fin.Rotation);
		return true;
	}
	return false;
}

bool TrackedObjectControl::UpdateFromLeapHand(double dt)
{
	if (!m_pRigid) return false;
	auto frame = m_pLeap->Controller().frame();
	XMMATRIX world = m_pLeap->ToWorldTransform();

	auto& hands = frame.hands();
	for (auto& hand : hands)
	{
		if (m_idx && hand.isRight() || !m_idx && hand.isLeft())
		{
			XMVECTOR pos = hand.palmPosition().toVector3<Vector3>();
			pos = XMVector3TransformCoord(pos, world);

			XMVECTOR xdir = hand.palmNormal().toVector3<Vector3>();
			XMVECTOR zdir = -hand.direction().toVector3<Vector3>();
			XMVECTOR ydir = XMVector3Cross(zdir, xdir);

			XMMATRIX rot = XMMatrixIdentity();
			rot.r[0] = xdir;
			rot.r[1] = ydir;
			rot.r[2] = zdir;

			//! HACK, since rot(world) == I
			XMVECTOR rotQ = XMQuaternionRotationMatrix(rot);

			m_pRigid->SetPosition(pos);
			m_pRigid->SetOrientation(rotQ);
			return true;
		}
	}
	return false;
}

TrackedObjectControl::TrackedObjectControl()
	:m_intrinsic(m_Transform.LocalTransform())
{
	m_pRigid = nullptr;
	m_pLeap = LeapMotion::GetForCurrentView();
	m_pVicon = IViconClient::GetFroCurrentView();

	if (m_pVicon && !m_pVicon->IsStreaming())
		m_pVicon.reset();

	XMMATRIX world = XMMatrixTranslation(0, 0.50f, 0.0f);
	m_pLeap->SetDeviceWorldCoord(world);

	m_posFilter.SetUpdateFrequency(&m_freq);
	m_rotFilter.SetUpdateFrequency(&m_freq);
}

TrackedObjectControl::~TrackedObjectControl()
{
}

void TrackedObjectControl::Parse(const ParamArchive * archive)
{
	SceneObject::Parse(archive);
	GetParam(archive, "index", m_idx);

	GetParam(archive, "translation", m_intrinsic.Translation);
	GetParam(archive, "scale", m_intrinsic.Scale);
	string rotstr;
	if (GetParam(archive, "rotation", rotstr))
	{
		if (rotstr[0] == '[')
		{
			Matrix4x4 mat = Matrix4x4::Identity;
			sscanf_s(rotstr.c_str(), "[%f,%f,%f;%f,%f,%f;%f,%f,%f]",
				&mat(0, 0), &mat(0, 1), &mat(0, 2),
				&mat(1, 0), &mat(1, 1), &mat(1, 2),
				&mat(2, 0), &mat(2, 1), &mat(2, 2)
				);
			auto m = XMLoad(mat);
			m = XMMatrixRotationRollPitchYaw(0, XMConvertToRadians(141), 0) * m;

			//XMMATRIX v = XMMatrixIdentity();
			//v *= m;
			auto q = XMQuaternionRotationMatrix(m);
			m_intrinsic.Rotation = q;
		}
		else if (rotstr[0] == '{') // Quaternion
		{
			Quaternion q;
			sscanf_s(rotstr.c_str(), "{%f,%f,%f,%f}", &q.x, &q.y, &q.z, &q.w);
			m_intrinsic.Rotation = q;
		}
		else if (rotstr[0] == '<')
		{
			Vector3 e;
			sscanf_s(rotstr.c_str(), "<%f,%f,%f>", &e.x, &e.y, &e.z);
			//m_intrinsic.Rotation = q;
		}
	}

	float tvcap = 2.0f, rvcap = 5.0f;
	GetParam(archive,"tvcap", tvcap);
	m_posFilter.SetCutoffFrequency(tvcap);
	GetParam(archive, "rvcap", rvcap);
	m_rotFilter.SetCutoffFrequency(rvcap);
}

void TrackedObjectControl::OnParentChanged(SceneObject * oldParent)
{
	m_pRigid = parent();
}

void TrackedObjectControl::Update(time_seconds const & time_delta)
{
	SceneObject::Update(time_delta);
	if (m_pVicon)
		UpdateFromVicon(time_delta.count());
	else if (m_pLeap)
		UpdateFromLeapHand(time_delta.count());
}
