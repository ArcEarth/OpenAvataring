#pragma once
#include "Math3D.h"

namespace Leap
{
	class Controller;
};

namespace Causality
{
	class TrackedBody;

	enum CursorButtonEnum
	{
		LButton,
		MButton,
		RButton,
		NoButton,
	};

	// Interface to register Application events
	class IAppComponent abstract
	{
	public:
		virtual ~IAppComponent();

	protected:
		void Register();
		void Unregister();

	public:
		// Retrive the given interface
		template <class T>
		const T* As() const 
		{ return dynamic_cast<const T*>(this); }
		// Retrive the given interface
		template <class T>
		T* As() { return dynamic_cast<T*>(this); }

		//const DirectX::Scene::IViewable* AsViewable() const { return dynamic_cast<const DirectX::Scene::IViewable*>(this); }
		//DirectX::Scene::IViewable* AsViewable() { return dynamic_cast<DirectX::Scene::IViewable*>(this); }
		//const DirectX::Scene::ITimeAnimatable* AsTimeAnimatedable() const { return dynamic_cast<const DirectX::Scene::ITimeAnimatable*>(this); }
		//DirectX::Scene::ITimeAnimatable* AsTimeAnimatedable() { return dynamic_cast<DirectX::Scene::ITimeAnimatable*>(this); }
		//const DirectX::Scene::IRenderable* AsRenderable() const { return dynamic_cast<const DirectX::Scene::IRenderable*>(this); }
		//DirectX::Scene::IRenderable* AsRenderable() { return dynamic_cast<DirectX::Scene::IRenderable*>(this); }

		//const Platform::ICursorInteractive* AsCursorInteractive() const { return dynamic_cast<const Platform::ICursorInteractive*>(this); }
		//ICursorInteractive* AsCursorInteractive() { return dynamic_cast<Platform::ICursorInteractive*>(this); }
		//const Platform::IKeybordInteractive* AsKeybordInteractive() const { return dynamic_cast<const Platform::IKeybordInteractive*>(this); }
		//IKeybordInteractive* AsKeybordInteractive() { return dynamic_cast<Platform::IKeybordInteractive*>(this); }
		//const Platform::IUserHandsInteractive* AsUserHandsInteractive() const { return dynamic_cast<const Platform::IUserHandsInteractive*>(this); }
		//IUserHandsInteractive* AsUserHandsInteractive() { return dynamic_cast<Platform::IUserHandsInteractive*>(this); }
	};

	struct CursorMoveEventArgs
	{
		// Relative position to window's top left corner
		Vector2 Position;
		Vector2 PositionDelta;
		float WheelDelta;
	};

	class ICursorController abstract
	{
		virtual Vector2 CurrentPosition() const = 0;
		virtual Vector2 DeltaPosition() const = 0;
		virtual bool IsButtonDown(CursorButtonEnum button) const = 0;
		virtual void SetCursorPosition(const Vector2& pos) = 0;
	};

	struct CursorButtonEvent
	{
		CursorButtonEvent() {}
		CursorButtonEvent(const CursorButtonEnum button)
			: Button (button)
		{}
		CursorButtonEnum Button;
	};

	class ICursorInteractive abstract
	{
	public:
		virtual void OnMouseButtonDown(const CursorButtonEvent &e) = 0;
		virtual void OnMouseButtonUp(const CursorButtonEvent &e) = 0;
		virtual void OnMouseMove(const CursorMoveEventArgs &e) = 0;
	};

	enum KeyModifiers
	{
		Mod_Shift = 0x001,
		Mod_Control = 0x002,
		Mod_Meta = 0x004,
		Mod_Alt = 0x008,
	};

	struct KeyboardEventArgs
	{
		unsigned Modifier;
		unsigned Key;
	};

	class IKeybordInteractive abstract
	{
	public:
		virtual void OnKeyDown(const KeyboardEventArgs&e) = 0;
		virtual void OnKeyUp(const KeyboardEventArgs&e) = 0;
	};

	class IJoyStickInteractive abstract
	{
	public:
	};

	struct UserHandsEventArgs
	{
		const Leap::Controller& sender;
		Matrix4x4 toWorldTransform;
	};

	class IUserPoseInteractive abstract
	{
	public:
		virtual void OnPoseChanged(TrackedBody*);
	};
}