all: ShadowMapGen ShadowMapEffect PostProcessingEffect BlurEffect BinaryShadowMapEffect

ShadowMapGen: ShadowMapGen.hlsl
	CompileShaders ShadowMapGen $(fxconfig)

ShadowMapEffect: ShadowMapEffectVS ShadowMapEffectPS

ShadowMapEffectVS: ShadowMapEffectVS.hlsl
	CompileShaders ShadowMapEffectVS $(fxconfig)

ShadowMapEffectPS: ShadowMapEffectPS.hlsl
	CompileShaders ShadowMapEffectPS $(fxconfig)

BinaryShadowMapEffect: ShadowMapEffectVS.hlsl ShadowMapEffectPS.hlsl
	CompileShaders BinaryShadowMapEffect $(fxconfig)
 
PostProcessingEffect: QuadVertexShader.hlsl
	CompileShaders QuadVertexShader $(fxconfig)

BlurEffect: BlurEffect.hlsl
	CompileShaders BlurEffect $(fxconfig)

ShadowMapEffectVS.hlsl ShadowMapEffectPS.hlsl: ShadowMapEffectStructures.hlsli ShadowMapEffectCBuffer.hlsli

clean:
	rmdir /S /Q windows
	rmdir /S /Q xbox