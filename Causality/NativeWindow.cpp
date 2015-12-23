#include "pch_bcl.h"
#include "NativeWindow.h"
#include "resource.h"

namespace Causality
{

	std::map<HWND, std::weak_ptr<IWindow>> Application::WindowsLookup;
	std::unique_ptr<Application> Application::Current;
	//Window::Window()
	//{
	//}

	//Window::~Window()
	//{
	//}

	NativeWindow::NativeWindow()
	{
	}

	void NativeWindow::Hide()
	{}

	void NativeWindow::Minimize() {}
	void NativeWindow::Maximize() {}

	bool NativeWindow::IsFullScreen() const {
		return m_FullScreen;
	}

	void NativeWindow::EnterFullScreen() {}
	void NativeWindow::ExitFullScreen() {}

	void NativeWindow::OnMouseMove(int x, int y)
	{
		auto current = Vector2((float)x, (float)y);
		CursorPositionDelta = current - CursorPostiton;
		CursorPostiton = current;
		if (!CursorMove.empty())
		{
			CursorMoveEventArgs e{
				CursorPostiton,
				CursorPositionDelta,
				WheelDelta, };
			CursorMove(e);
		}
	}

	inline void NativeWindow::OnKeyDown(unsigned char key)
	{
		switch (key)
		{
		case VK_LBUTTON:
			if (ButtonStates[LButton]) return;
			ButtonStates[LButton] = true;
			CursorButtonDown(CursorButtonEvent(LButton));
			break;
		case VK_RBUTTON:
			if (ButtonStates[RButton]) return;
			ButtonStates[RButton] = true;
			CursorButtonDown(CursorButtonEvent(RButton));
			break;
		case VK_MBUTTON:
			if (ButtonStates[MButton]) return;
			ButtonStates[MButton] = true;
			CursorButtonDown(CursorButtonEvent(MButton));
			break;
		default:
			if (Keys[key]) return;
			Keys[key] = true;
			KeyDown(KeyboardEventArgs{ GetCurrentModifiers(),key });
			break;
		}
	}

	inline void NativeWindow::OnKeyUp(unsigned char key) {
		switch (key)
		{
		case VK_LBUTTON:
			if (!ButtonStates[LButton]) return;
			ButtonStates[LButton] = false;
			CursorButtonUp(CursorButtonEvent(LButton));
			break;
		case VK_RBUTTON:
			if (!ButtonStates[RButton]) return;
			ButtonStates[RButton] = false;
			CursorButtonUp(CursorButtonEvent(RButton));
			break;
		case VK_MBUTTON:
			if (!ButtonStates[MButton]) return;
			ButtonStates[MButton] = false;
			CursorButtonUp(CursorButtonEvent(MButton));
			break;
		default:
			if (!Keys[key]) return;
			Keys[key] = false;
			KeyUp(KeyboardEventArgs{ GetCurrentModifiers(),key });
			break;
		}
	}

	void NativeWindow::OnResize(size_t width, size_t height)
	{
		if (!SizeChanged.empty())
		{
			SizeChanged(Vector2(width, height));
		}
	}

	NativeWindow::~NativeWindow()
	{
		Close();
	}

	void NativeWindow::Initialize(const std::string& title, unsigned int screenWidth, unsigned int screenHeight, bool fullScreen)
	{
		WNDCLASSEX wc;
		DEVMODE dmScreenSettings;
		int posX, posY;

		// Get the instance of this application.
		m_hInstance = GetModuleHandle(NULL);

		// Give the application a name.
		m_Title.assign(title.begin(),title.end());

		// Setup the windows class with default settings.
		wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
		wc.lpfnWndProc = Application::WndProc;
		wc.cbClsExtra = 0;
		wc.cbWndExtra = 0;
		wc.hInstance = m_hInstance;
		wc.hIcon = LoadIcon(NULL, MAKEINTRESOURCE(IDI_ICON1));
		wc.hIconSm = wc.hIcon;
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);
		wc.hbrBackground = (HBRUSH) GetStockObject(WHITE_BRUSH);
		wc.lpszMenuName = NULL;
		wc.lpszClassName = m_Title.c_str();
		wc.cbSize = sizeof(WNDCLASSEX);

		// Register the window class.
		RegisterClassEx(&wc);

		// Setup the screen settings depending on whether it is running in full screen or in windowed mode.
		if (fullScreen)
		{
			// If full screen set the screen to maximum size of the users desktop and 32bit.
			// Determine the resolution of the clients desktop screen.
			screenHeight = GetSystemMetrics(SM_CYSCREEN);
			screenWidth = GetSystemMetrics(SM_CXSCREEN);

			memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
			dmScreenSettings.dmSize = sizeof(dmScreenSettings);
			dmScreenSettings.dmPelsWidth = (unsigned long) screenWidth;
			dmScreenSettings.dmPelsHeight = (unsigned long) screenHeight;
			dmScreenSettings.dmBitsPerPel = 32;
			dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

			// Change the display settings to full screen.
			ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN);

			// Set the position of the window to the top left corner.
			posX = posY = 0;
		}
		else
		{
			posX = (GetSystemMetrics(SM_CXSCREEN) - (int)screenWidth) / 2;
			posY = (GetSystemMetrics(SM_CYSCREEN) - (int)screenHeight) / 2;
		}

		// Create the window with the screen settings and get the handle to it.
		m_hWnd = CreateWindowEx(WS_EX_APPWINDOW, wc.lpszClassName, wc.lpszClassName,
			WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_THICKFRAME,
			posX, posY, screenWidth, screenHeight, NULL, NULL, m_hInstance, NULL);

		Application::WindowsLookup[m_hWnd] = this->shared_from_this();

		// Bring the window up on the screen and set it as main focus.
		ShowWindow(m_hWnd, SW_SHOW);
		SetForegroundWindow(m_hWnd);
		SetFocus(m_hWnd);

		RECT bound;
		GetWindowRect(m_hWnd, &bound);
		m_Boundary.Position.x = (float)bound.left;
		m_Boundary.Position.y = (float) bound.top;
		m_Boundary.Size.x = (float) (bound.right - bound.left);
		m_Boundary.Size.y = (float) (bound.bottom - bound.top);

		return;
	}

	void NativeWindow::Show()
	{
		ShowWindow(m_hWnd, SW_SHOW);
	}

	void NativeWindow::Focus()
	{
		SetForegroundWindow(m_hWnd);
		SetFocus(m_hWnd);
	}

	void NativeWindow::Close()
	{
		//// Show the mouse cursor.
		//ShowCursor(true);

		// Fix the display settings if leaving full screen mode.
		if (m_FullScreen)
		{
			ChangeDisplaySettings(NULL, 0);
		}

		// Remove the window.
		DestroyWindow(m_hWnd);

		m_hWnd = NULL;

		// Remove the application instance.
		UnregisterClass(m_Title.c_str(), m_hInstance);
		m_hInstance = NULL;

		return;
	}


	LRESULT CALLBACK Application::WndProc(HWND hwnd, UINT umessage, WPARAM wparam, LPARAM lparam)
	{
		//return Application::CoreWindow->MessageHandler(umessage, wparam, lparam);
		std::shared_ptr<IWindow> window = nullptr;
		auto itr = WindowsLookup.find(hwnd);
		if (itr != WindowsLookup.end() && !itr->second.expired())
			window = itr->second.lock();
		switch (umessage)
		{
			// Check if the window is being destroyed.
		case WM_DESTROY:
		{
			PostQuitMessage(0);
			Current->exitProposal = true;
			return S_OK;
		}
		break;

		case WM_SIZE:
		{
			window->OnResize(LOWORD(lparam), HIWORD(lparam));
		}
		break;
		// Check if the window is being closed.
		case WM_CLOSE:
		{
			if (window)
			{
				window->Close();
				PostQuitMessage(0);
			}
			return S_OK;
		}
		break;
		case WM_LBUTTONDOWN:
		case WM_RBUTTONDOWN:
		case WM_MBUTTONDOWN:
		{
			if (window)
			{
				auto key = (umessage - WM_LBUTTONDOWN) / 3U + 1U;
				if (key == 3) key++;
				window->OnKeyDown(key);
			}
		}
		break;
		case WM_LBUTTONUP:
		case WM_RBUTTONUP:
		case WM_MBUTTONUP:
		{
			if (window)
			{
				auto key = (umessage - WM_LBUTTONUP) / 3U + 1U;
				if (key == 3) key++;
				window->OnKeyUp(key);
			}
		}
		break;
		//Check if a key has been pressed on the keyboard.
		case WM_KEYDOWN:
		{
			// If a key is pressed send it to the input object so it can record that state.
			if (window) window->OnKeyDown((unsigned char) wparam);
			//m_pInput->KeyDown((unsigned int) wparam);
			return S_OK;
		}
		break;

		// Check if a key has been released on the keyboard.
		case WM_KEYUP:
		{
			// If a key is released then send it to the input object so it can unset the state for that key.
			if (window) window->OnKeyUp((unsigned char) wparam);
			//m_pInput->KeyUp((unsigned int) wparam);
			return S_OK;
		}
		break;

		//Handel the mouse(hand gesture glove) input
		case WM_MOUSEMOVE:
		{
			if (window) window->OnMouseMove(LOWORD(lparam), HIWORD(lparam));
			return S_OK;
		}
		break;
		//Any other messages send to the default message handler as our application won't make use of them.
		// All other messages pass to the message handler in the system class.
		default:
		{
			return DefWindowProc(hwnd, umessage, wparam, lparam);
		}

		}
		return S_OK;
	}
DirectX::Vector2 CursorHandler::CurrentPosition() const
{
	return CursorPostiton;
}
DirectX::Vector2 CursorHandler::DeltaPosition() const
{
	return CursorPositionDelta;
}
bool CursorHandler::IsButtonDown(CursorButtonEnum button) const
{
	return ButtonStates[button];
}
void CursorHandler::SetCursorPosition(const DirectX::Vector2 & pos)
{
}

void DebugConsole::Initialize(const std::string& title, unsigned int width, unsigned int height, bool fullScreen)
{
	if (!AllocConsole())
		return;
	HWND hWnd = GetConsoleWindow();
	MoveWindow(hWnd, 0, 0, width, height, TRUE);

#pragma warning( push )
#pragma warning( disable: 4996 )
	// Reopen file handles for stdin,stdout and sterr to point to
	// the newly created console
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

#pragma warning( pop )

	std::cout << "DEBUG console has been created" << std::endl;

	Application::WindowsLookup[hWnd] = shared_from_this();
	return;
}

void DebugConsole::Show()
{
}

void DebugConsole::Hide()
{
}

void DebugConsole::Focus()
{
}

void DebugConsole::Close()
{
}

void DebugConsole::Minimize()
{
}

void DebugConsole::Maximize()
{
}

bool DebugConsole::IsFullScreen() const
{
	return false;
}

void DebugConsole::EnterFullScreen()
{
}

void DebugConsole::ExitFullScreen()
{
}

void DebugConsole::OnMouseMove(int x, int y)
{
}

void DebugConsole::OnKeyDown(unsigned char key)
{
}

void DebugConsole::OnKeyUp(unsigned char key)
{
}

void DebugConsole::OnResize(size_t width, size_t height)
{
}

}