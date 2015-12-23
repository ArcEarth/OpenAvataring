#include "pch_bcl.h"
#include "BasicKeyboardMouseControlLogic.h"
#include "Scene.h"
#include "CausalityApplication.h"

using namespace Causality;

REGISTER_SCENE_OBJECT_IN_PARSER(first_person_keyboard_mouse_control, KeyboardMouseFirstPersonControl);

KeyboardMouseFirstPersonControl::KeyboardMouseFirstPersonControl(IRigid* pTarget)
{
	Speed = 2.0f;
	SetTarget(pTarget);
}

void KeyboardMouseFirstPersonControl::SetTarget(IRigid * pTarget)
{
	if (!pTarget && m_pTarget)
	{
		Unregister();
	}
	else if (pTarget)
	{
		bool needToReg = !m_pTarget;
		m_pTarget = pTarget;
		InitialOrientation = pTarget->GetOrientation();
		AddationalYaw = 0;
		AddationalPitch = 0;
		AddationalRoll = 0;
		if (needToReg)
			Register();
	}

}

void KeyboardMouseFirstPersonControl::OnParentChanged(SceneObject * oldParent)
{
	SetTarget(parent());
}

void KeyboardMouseFirstPersonControl::Update(time_seconds const& time_delta)
{
	using namespace DirectX;
	if (m_pTarget && CameraVeclocity.LengthSquared() > 0.01f)
	{
		XMVECTOR disp = XMVector3Normalize(CameraVeclocity);
		disp = XMVector3Rotate(disp, m_pTarget->GetOrientation());
		disp *= Speed * (float)time_delta.count();
		m_pTarget->Move(disp);
	}
}

void KeyboardMouseFirstPersonControl::OnKeyDown(const KeyboardEventArgs & e)
{
	if (!m_pTarget) return;

	if (e.Key == 'W')
		CameraVeclocity += Vector3{ 0,0,-1 };
	if (e.Key == 'S')
		CameraVeclocity += Vector3{ 0,0,1 };
	if (e.Key == 'A')
		CameraVeclocity += Vector3{ -1,0,0 };
	if (e.Key == 'D')
		CameraVeclocity += Vector3{ 1,0,0 };
}

void KeyboardMouseFirstPersonControl::OnKeyUp(const KeyboardEventArgs & e)
{
	if (!m_pTarget) return;
	if (e.Key == 'W')
		CameraVeclocity -= Vector3{ 0,0,-1 };
	if (e.Key == 'S')
		CameraVeclocity -= Vector3{ 0,0,1 };
	if (e.Key == 'A')
		CameraVeclocity -= Vector3{ -1,0,0 };
	if (e.Key == 'D')
		CameraVeclocity -= Vector3{ 1,0,0 };
	if (e.Key == 'K')
		g_DebugView = !g_DebugView;
	if (e.Key == 'T')
		g_ShowCharacterMesh = !g_ShowCharacterMesh;

	if (e.Key == '-' || e.Key == '_' || e.Key == VK_SUBTRACT || e.Key == VK_OEM_MINUS)
		this->Scene->SetTimeScale(this->Scene->GetTimeScale() - 0.1);

	if (e.Key == '=' || e.Key == '+' || e.Key == VK_ADD || e.Key == VK_OEM_PLUS)
		this->Scene->SetTimeScale(this->Scene->GetTimeScale() + 0.1);

	if (e.Key == '0' || e.Key == ')')
		this->Scene->SetTimeScale(1.0);

	if (e.Key == VK_SPACE)
	{
		if (!this->Scene->IsPaused())
			this->Scene->Pause();
		else
			this->Scene->Resume();
	}

	if (e.Key == VK_ESCAPE)
		App::Current()->Exit();

	if (e.Key == VK_OEM_4) // [{
	{
		g_DebugArmatureThinkness = std::max(0.001f, g_DebugArmatureThinkness - 0.002f);
	}
	if (e.Key == VK_OEM_6) // ]}
	{
		g_DebugArmatureThinkness = std::min(0.015f, g_DebugArmatureThinkness + 0.002f);
	}

}

void KeyboardMouseFirstPersonControl::OnMouseButtonDown(const CursorButtonEvent & e)
{
	switch (e.Button)
	{
	case CursorButtonEnum::LButton:
		leftButtonDown = true;
		break;
	case CursorButtonEnum::RButton:
		rightButtonDown = true;
		break;
	case CursorButtonEnum::MButton:
		IsTrackingCursor = true;
		//CursorMoveEventConnection = pWindow->CursorMove += MakeEventHandler(&App::OnCursorMove_RotateCamera, this);
		break;
	default:
		break;
	}
	//if (e.Button == CursorButtonEnum::MButton)
	//{
	//	IsTrackingCursor = true;
	//	rightButtonDown = true;
	//}
	//else {
	//	leftButtonDown = true;
	//}
}

void KeyboardMouseFirstPersonControl::OnMouseButtonUp(const CursorButtonEvent & e)
{
	switch (e.Button)
	{
	case CursorButtonEnum::LButton:
		leftButtonDown = false;
		break;
	case CursorButtonEnum::RButton:
		rightButtonDown = false;
		break;
	case CursorButtonEnum::MButton:
		IsTrackingCursor = false;
		break;
	default:
		break;
	}
}

void KeyboardMouseFirstPersonControl::OnMouseMove(const CursorMoveEventArgs & e)
{
	using namespace DirectX;
	if (!IsTrackingCursor) return;
	auto yaw = -e.PositionDelta.x / 1000.0f * XM_PI;
	auto pitch = -e.PositionDelta.y / 1000.0f * XM_PI;
	AddationalYaw += yaw;
	AddationalPitch += pitch;
	if (m_pTarget)
	{
		XMVECTOR extrinsic = XMQuaternionRotationRollPitchYaw(AddationalPitch, AddationalYaw, 0);
		XMVECTOR intial = InitialOrientation;
		intial = XMQuaternionMultiply(intial, extrinsic);
		m_pTarget->SetOrientation(intial);
	}
}
