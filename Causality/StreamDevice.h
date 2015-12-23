#pragma once
#include "Serialization.h"

namespace Causality
{
	namespace Devices
	{
		class IStreamDevice abstract
		{
		public:
			virtual ~IStreamDevice();

			virtual bool Initialize(const ParamArchive * archive) = 0;

			virtual bool Start() = 0;
			virtual void Stop() = 0;
			virtual bool IsStreaming() = 0;

			virtual bool Update() = 0;
		};
	}
}