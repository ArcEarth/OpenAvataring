#pragma once
#include "Serialization.h"
#include <memory>

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

			virtual bool Update() = 0;

			virtual bool IsStreaming() const = 0;
			virtual bool IsAsychronize() const = 0;
		};
	}
}