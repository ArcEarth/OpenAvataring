#pragma once
#include "BCL.h"
#include <memory>
#include <Textures.h>
#include "DeviceResources.h"
#include <exception>

namespace Causality
{
	enum EyesEnum : int
	{
		Eye_Left = 0,
		Eye_Right = 1,
		Eye_Count = 2,
	};

	struct StaticPose
	{
	public:
		Quaternion	Orientation;
		Vector3		Position;
	};

	struct DynamicPose
	{
		Quaternion	Orientation;
		Vector3		Position;
		Vector3		AngularVelocity;
		Vector3		Velocity;
		Vector3		AngularAcceleration;
		Vector3		Acceleration;
		double		TimeInSeconds;         // Absolute time of this state sample.
	};

	namespace Devices
	{

		class device_not_exist : public std::runtime_error
		{

		};

		class OculusRift
		{
		public:

			OculusRift();
			~OculusRift();

			static std::shared_ptr<OculusRift> GetForCurrentView();

			static std::weak_ptr<OculusRift> wpCurrentDevice;

			static std::shared_ptr<OculusRift> Create(int hmdIdx = 0);

			static bool Initialize(void);

			bool InitializeGraphics(HWND hWnd, DirectX::DeviceResources* pDeviceResource);

			DirectX::Vector2 Resoulution() const;
			DirectX::Vector2 DesktopWindowPosition() const;
			const char* DisplayDeviceName() const;

			// Rendering Methods and Properties
			void DissmisHealthWarnning();
			void BeginFrame();
			void EndFrame();
			void SetView(EyesEnum eye);

			DirectX::RenderTarget& ViewTarget(EyesEnum eye);
			DirectX::RenderableTexture2D& ColorBuffer();
			DirectX::DepthStencilBuffer& DepthStencilBuffer();

			DirectX::XMMATRIX EyeProjection(EyesEnum eye) const;

			// Tracking States
			const StaticPose& EyePoses(EyesEnum eye) const;
			float UserEyeHeight() const;
			const DynamicPose& HeadPose() const;

		private:
			class Impl;
			std::unique_ptr<Impl> pImpl;
		};

		enum GenderEnum
		{
			Unspecifed,
			Male,
			Female,
		};

		struct UserProfile
		{
			int UserIdx;
			std::string Name;
			GenderEnum Gender;
			float PlayerHeight;
			float EyeHeight;
			float IPD;
			float NeckEyeDistance;
			float CenteredFromWorld;
		};

		//class IPositionSensor abstract
		//{
		//public:
		//	virtual Math::Vector3 CurrentPosition() = 0;
		//	virtual Math::Vector3 CurrentVelocity() = 0;
		//	event Windows::Foundation::EventHandler<Platform::Object^>^ PositionChanged;
		//};
		//
		//interface class IOrientationSensor 
		//{
		//public:
		//	property Math::Quaternion CurrentOrientation { Math::Quaternion get(); };
		//	event Windows::Foundation::EventHandler<Platform::Object^>^ OrientationChanged;
		//};
		//
		//interface class IHeadMountedDisplay  : public IPositionSensor, public IOrientationSensor
		//{
		//public:
		//	property float LogicalDpi;
		//	property Windows::Foundation::Size DisplaySize { Windows::Foundation::Size get(); }
		//	property Windows::Foundation::Size EyeViewSize { Windows::Foundation::Size get(); }
		//	property bool NeedDisortion { bool get(); };
		//	virtual Windows::Foundation::Rect EyeViewport(int eye) = 0;
		//	virtual Math::Matrix4 EyeView(int eye) const = 0;
		//	virtual EyePose EyePose(int eye) const = 0;
		//	virtual Math::Matrix4 EyeProjection(int eye) = 0;
		//};

	}
}