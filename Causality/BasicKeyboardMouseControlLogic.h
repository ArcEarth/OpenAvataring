#pragma once
#include "SceneObject.h"
#include "Interactive.h"

namespace Causality
{
	extern bool		g_DebugView;
	extern bool		g_ShowCharacterMesh;
	extern float	g_DebugArmatureThinkness;
	extern bool		g_MirrowInputX;

	class KeyboardMouseFirstPersonControl : public SceneObject, public IAppComponent, public IKeybordInteractive, public ICursorInteractive
	{
	public:
		KeyboardMouseFirstPersonControl(IRigid* pTarget = nullptr);

		void SetTarget(IRigid* pTarget);

		virtual void OnParentChanged(SceneObject* oldParent) override;
		// Inherited via ITimeAnimatable
		virtual void Update(time_seconds const& time_delta) override;

		// Inherited via IKeybordInteractive
		virtual void OnKeyDown(const KeyboardEventArgs & e) override;
		virtual void OnKeyUp(const KeyboardEventArgs & e) override;

		// Inherited via ICursorInteractive
		virtual void OnMouseButtonDown(const CursorButtonEvent & e) override;
		virtual void OnMouseButtonUp(const CursorButtonEvent & e) override;
		virtual void OnMouseMove(const CursorMoveEventArgs & e) override;

		bool isLeftButtonDown() const { return leftButtonDown;  }
		bool isRightButtonDown() const { return rightButtonDown; }
	public:
		float											Speed;
		float											AngularSpeed;

		DirectX::Quaternion								InitialOrientation;

		float											AddationalYaw = 0;
		float											AddationalPitch = 0;
		float											AddationalRoll = 0;

	private:
		IRigid*											m_pTarget = nullptr;

		bool											IsTrackingCursor = false;
		bool											leftButtonDown = false;
		bool											rightButtonDown = false;
		DirectX::Vector3								CameraVeclocity;
		DirectX::Vector3								CameraAngularVeclocity;
		DirectX::Vector3								AngularDisplacement; // Eular Angle
	};
}