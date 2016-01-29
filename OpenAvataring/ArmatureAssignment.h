#pragma once
#include "CharacterController.h"

namespace Causality
{
	struct CtrlTransformInfo
	{
		float likilihood;
		string clipname;
		unique_ptr<ArmatureTransform> transform;
	};

	CtrlTransformInfo
		CreateControlTransform(const CharacterController & controller, const ClipFacade& iclip);
}