#pragma once
#include "Interactive.h"
#include "Events.h"
#include <string>
#define NOMINMAX
#include <minwindef.h>

namespace Causality
{
	using DirectX::Vector2;

	struct Rect
	{
		Vector2 Position;
		Vector2 Size;

		float Top() const { return Position.y; }
		float Left() const { return Position.x; }
		float Bottom() const { return Position.y + Size.y; }
		float Right() const { return Position.x + Size.x; }
		float Width() const { return Size.x; }
		float Height() const { return Size.y; }
	};

	//ref class Window;
	class NativeWindow;

	class IWindow;

	class Application
	{
	public:
		template <class TDerived>
		static int Invoke(const std::vector<std::string>& args)
		{
			Current = std::make_unique<TDerived>();
			return Current->Run(args);
		}

	public:
		Application();

		virtual ~Application();

		int Run(const std::vector<std::string>& args);

		void Exit();

		virtual bool OnStartup(const std::vector<std::string>& args) = 0;
		virtual void OnExit() = 0;
		virtual bool OnIdle() = 0;

		HINSTANCE Instance()
		{
			return hInstance;
		}

	public:
		static std::unique_ptr<Application> Current;
	public:
		static std::map<HWND, std::weak_ptr<IWindow>> WindowsLookup;
		static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
	protected:
		HINSTANCE	hInstance;
		bool		exitProposal = false;

	};

	class IWindow abstract
	{
	public:
		~IWindow() {}

		virtual HWND Handle() const = 0;
		virtual void Initialize(const std::string& title, unsigned int width, unsigned int height, bool fullScreen = false) = 0;
		virtual void Show() = 0;
		virtual void Hide() = 0;
		virtual void Focus() = 0;
		virtual void Close() = 0;
		virtual void Minimize() = 0;
		virtual void Maximize() = 0;
		virtual bool IsFullScreen() const = 0;
		virtual void EnterFullScreen() = 0;
		virtual void ExitFullScreen() = 0;

		virtual void OnResize(size_t width, size_t height) = 0;
		virtual void OnMouseMove(int x, int y) = 0;
		virtual void OnKeyDown(unsigned char key) = 0;
		virtual void OnKeyUp(unsigned char key) = 0;
	};

	struct CursorHandler : public ICursorController
	{
	public:
		CursorHandler()
		{
			std::fill_n(ButtonStates, 3, false);
			WheelValue = 0;
			WheelDelta = 0;
		}
		// Event Interface
		Event<const CursorButtonEvent&> CursorButtonDown;
		Event<const CursorButtonEvent&> CursorButtonUp;
		Event<const CursorMoveEventArgs&> CursorMove;

		Vector2 CursorPostiton;
		Vector2 CursorPositionDelta;
		float	WheelValue;
		float	WheelDelta;
		bool	ButtonStates[3];

		// Inherited via ICursorController
		virtual DirectX::Vector2 CurrentPosition() const override;
		virtual DirectX::Vector2 DeltaPosition() const override;
		virtual bool IsButtonDown(CursorButtonEnum button) const override;
		virtual void SetCursorPosition(const DirectX::Vector2 & pos) override;
	};

	struct KeyboardHandler
	{
	public:
		KeyboardHandler()
		{
			memset(Keys, 0, 255 * sizeof(BOOL));
		}
		unsigned GetCurrentModifiers() const
		{
			return (Keys[VK_CONTROL] & Mod_Control) | (Keys[VK_SHIFT] & Mod_Shift) | (Keys[VK_MENU] & Mod_Alt) | ((Keys[VK_LWIN] | Keys[VK_RWIN])& Mod_Meta);
		}

		Event<const KeyboardEventArgs&> KeyDown;
		Event<const KeyboardEventArgs&> KeyUp;
		BOOL	Keys[255];
	};

	class DebugConsole : public std::enable_shared_from_this<DebugConsole>, public IWindow
	{
	public:
		HWND Handle() const { return hWnd; }

		DebugConsole()
			: hWnd(NULL)
		{}
		// Inherited via IWindow
		virtual void Initialize(const std::string& title, unsigned int width, unsigned int height, bool fullScreen = false) override;
		virtual void Show() override;
		virtual void Hide() override;
		virtual void Focus() override;
		virtual void Close() override;
		virtual void Minimize() override;
		virtual void Maximize() override;
		virtual bool IsFullScreen() const override;
		virtual void EnterFullScreen() override;
		virtual void ExitFullScreen() override;
		virtual void OnMouseMove(int x, int y) override;
		virtual void OnKeyDown(unsigned char key) override;
		virtual void OnKeyUp(unsigned char key) override;
		virtual void OnResize(size_t width, size_t height) override;


	private:
		HWND hWnd;

	};

	// Design to work with std::shared_ptr<NativeWindow>
	class NativeWindow :public std::enable_shared_from_this<NativeWindow>,  public IWindow, public CursorHandler, public KeyboardHandler
	{
	public:
		NativeWindow();
		~NativeWindow();
		virtual void Initialize(const std::string& title, unsigned int width, unsigned int height, bool fullScreen = false) override;

		void Show();
		void Hide();
		void Focus();

		void Close();

		void Minimize();
		void Maximize();

		bool IsFullScreen() const;

		void EnterFullScreen();
		void ExitFullScreen();

		virtual void OnMouseMove(int x, int y) override;
		virtual void OnKeyDown(unsigned char key) override;
		virtual void OnKeyUp(unsigned char key) override;
		virtual void OnResize(size_t width, size_t height) override;

		HWND Handle() const
		{
			return m_hWnd;
		}

		HINSTANCE ApplicationInstance() const
		{
			return m_hInstance;
		}

		Rect Boundary() const
		{
			return m_Boundary;
		}

		Event<const Vector2&> SizeChanged;

	private:
		std::wstring		m_Title;
		HWND				m_hWnd;
		HINSTANCE			m_hInstance;
		bool				m_FullScreen;
		Rect				m_Boundary;
	};

	//ref class Window sealed
	//{
	//public:
	//	Window();
	//	//Window(std::unique_ptr<NativeWindow> &&native);
	//	virtual ~Window();
	//	void Initialize(Platform::String^ title, unsigned int screenWidth, unsigned int screenHeight, bool fullScreen = false);
	//	void Show();
	//	void Focus();
	//	void Close();

	//	event Windows::Foundation::TypedEventHandler<Window^, Windows::UI::Core::WindowActivatedEventArgs^> ^Activated{
	//		Windows::Foundation::EventRegistrationToken add(Windows::Foundation::TypedEventHandler<Window^, Windows::UI::Core::WindowActivatedEventArgs^>^ value);
	//		void remove(Windows::Foundation::EventRegistrationToken token);
	//		void raise(Window^, Windows::UI::Core::WindowActivatedEventArgs^);
	//	}

	//private:
	//	std::unique_ptr<NativeWindow>	m_Native;
	//};

}
