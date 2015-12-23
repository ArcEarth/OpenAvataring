#pragma once
#include "SmartPointers.h"
#include "String.h"
#include "Math3D.h"
#include "StreamDevice.h"

namespace Causality
{
	namespace Devices
	{
		class IViconClient : public IStreamDevice
		{
		public:
			static sptr<IViconClient> GetFroCurrentView();
			static sptr<IViconClient> Create(const std::string &serverIP = "");
			virtual ~IViconClient();

			virtual const IsometricTransform& GetWorldTransform() = 0;
			virtual void SetWorldTransform(const IsometricTransform& transform) = 0;

			virtual RigidTransform GetRigid(int index) = 0;
			virtual RigidTransform GetRigid(const string& name) = 0;
			virtual int GetRigidID(const string & name) = 0;

			// Inherited via IStreamDevice
			virtual bool Start() = 0;
			virtual void Stop() = 0;
			virtual bool Update() = 0;
			virtual bool Initialize(const ParamArchive * archive) = 0;
			virtual bool IsStreaming() = 0;
			virtual void SetVerbose(bool verbose) = 0;
		};
	}
}