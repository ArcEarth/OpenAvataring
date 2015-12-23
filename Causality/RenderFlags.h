#pragma once
namespace Causality
{
	enum RenderFlagPrimtive : unsigned int
	{
		Visible = 0,
		RecivesLight,
		DropsShadow,
		RecivesShadow,
		Reflective,
		Refrective,
		LightSource,
		AcceptCustomizeEffects,
		Skinable,
		RequireBloomEffect,
		FinalPassOnly,
	};

	#include "CompositeFlag.h"

	class RenderFlags : public CompositeFlag<RenderFlagPrimtive>
	{
	public:
		typedef CompositeFlag<RenderFlagPrimtive> Base;
		using Base::CompositeFlag;
		RenderFlags(unsigned flags) : Base(flags) {}

		static const unsigned
			OpaqueObjects = 1 << Visible | 1 << RecivesLight | 1 << DropsShadow | 1 << RecivesShadow | 1 << AcceptCustomizeEffects,
			SemiTransparentObjects = 1 << Visible | 1 << RecivesLight | 1 << DropsShadow | 1 << AcceptCustomizeEffects,
			GhostObjects = 1 << Visible | 1 << RecivesLight | 1 << AcceptCustomizeEffects,
			SpecialEffects = 1 << Visible | 1 << FinalPassOnly,
			SkyView = 1 << Visible,
			Lights = 1 << LightSource,
			Visible = 1 << Visible,
			RecivesLight = 1 << RecivesLight,
			DropsShadow = 1 << DropsShadow,
			RecivesShadow = 1 << RecivesShadow,
			Reflective = 1 << Reflective,
			Refrective = 1 << Refrective,
			LightSource = 1 << LightSource,
			Skinable = 1 << Skinable,
			AcceptCustomizeEffects = 1 << AcceptCustomizeEffects,
			BloomEffectSource = Visible | 1 << RequireBloomEffect | AcceptCustomizeEffects;
	};


}