#include "pch_bcl.h"
#include "CharacterObject.h"
#include <PrimitiveVisualizer.h>
#include <ShadowMapGenerationEffect.h>
#include "Settings.h"
#include <Models.h>
#include <ShaderEffect.h>
#include "Scene.h"
#include "AssetDictionary.h"

using namespace Causality;
using namespace DirectX;
using namespace DirectX::Scene;

REGISTER_SCENE_OBJECT_IN_PARSER(creature, CharacterObject);

bool g_DrawPartBoxes = false;
bool g_DrawXRayArmature = true;
double updateFrequency = 60;

void CharacterObject::DisplaceByVelocityFrame()
{
	static const float threshold = 0.05f;
	float frectionFactor = 1 / 0.05f;
	auto &frame = m_CurrentFrame;
	auto &vframe = m_VelocityFrame;
	auto &armature = *m_pArmature;

	if (vframe.size() == 0)
		return;

	auto world = this->GlobalTransformMatrix();

	XMVECTOR vsum = XMVectorZero();
	int count = 0;

	float lowest = std::numeric_limits<float>::max();
	for (int jid = 0; jid < armature.size(); jid++)
	{
		XMVECTOR pos = XMVector3Transform(XMLoadA(frame[jid].GblTranslation), world);
		float y = XMVectorGetY(pos);
		if (XMVectorGetY(pos) < threshold)
		{
			XMVECTOR vec = XMLoadA(vframe[jid].LinearVelocity);
			vec = XMVectorSetW(vec, 0.f);
			vec = XMVector4Transform(vec, world);

			float presure = sqrt((threshold - y) * frectionFactor);
			presure = std::min(presure, 2.0f);

			if (y < lowest)
				lowest = y;

			vsum += (vec * presure);
			count++;
		}
	}

	if (count != 0)
		vsum /= count;

	vsum = XMVectorAndInt(vsum, g_XMSelect1010);

	vsum *= -0.1f;
	float speed = XMVectorGetX(XMVector3Length(vsum));

	if (speed > 1e-5f)
	{
		vsum /= speed;

		speed = m_SpeedFilter.Apply(speed);

		if (speed > g_MaxCharacterSpeed)
			speed = g_MaxCharacterSpeed;

		vsum *= speed;
	}
	else
	{
		m_SpeedFilter.Apply(speed);
	}


	// make sure model is "Grounded"
	vsum -= lowest * g_XMIdentityR1;

	this->SetPosition((XMVECTOR)GetPosition() + vsum);
}

void CharacterObject::ComputeVelocityFrame(time_seconds time_delta)
{
	auto &cframe = m_CurrentFrame;
	auto &lframe = m_LastFrame;
	auto &vframe = m_VelocityFrame;
	auto &armature = *m_pArmature;

	if (lframe.size() == 0)
		return;

	if (vframe.size() == 0)
		vframe.resize(armature.size());

	for (size_t i = 0; i < armature.size(); i++)
	{
		XMVECTOR disp = XMLoadA(cframe[i].GblTranslation) - XMLoadA(lframe[i].GblTranslation);
		XMStoreA(vframe[i].LinearVelocity, disp / time_delta.count());
	}
}

const CharacterObject::frame_type & CharacterObject::GetCurrentFrame() const
{
	return m_CurrentFrame;
}

CharacterObject::frame_type & CharacterObject::MapCurrentFrameForUpdate()
{
	m_UpdateLock = true;
	m_LastFrame = m_CurrentFrame;
	return m_CurrentFrame;
}

void CharacterObject::ReleaseCurrentFrameFrorUpdate()
{
	m_UpdateLock = false;
}

IArmature & CharacterObject::Armature() { return *m_pArmature; }

const IArmature & CharacterObject::Armature() const { return *m_pArmature; }

BehavierSpace & CharacterObject::Behavier() { return *m_pBehavier; }

const BehavierSpace & CharacterObject::Behavier() const { return *m_pBehavier; }

void CharacterObject::SetBehavier(BehavierSpace & behaver) {
	m_pBehavier = &behaver;
	m_pArmature = &m_pBehavier->Armature();
	m_CurrentFrame.resize(m_pArmature->size());
}

const ArmatureFrameAnimation * CharacterObject::CurrentAction() const { return m_pCurrentAction; }

string CharacterObject::CurrentActionName() const { return m_pCurrentAction ? m_pCurrentAction->Name : ""; }

bool CharacterObject::StartAction(const string & key, time_seconds begin_time, bool loop, time_seconds transition_time)
{
	if (!m_pBehavier->Contains(key))
		return false;
	auto& anim = (*m_pBehavier)[key];
	m_pCurrentAction = &anim;
	m_CurrentActionTime = begin_time;
	m_LoopCurrentAction = loop;
	return true;
}

bool CharacterObject::StopAction(time_seconds transition_time)
{
	m_pCurrentAction = nullptr;
	m_LoopCurrentAction = false;
	return true;
}

const XMMATRIX * Causality::CharacterObject::GetBoneTransforms() const
{
	return reinterpret_cast<const XMMATRIX*>(m_BoneTransforms.data());
}

void CharacterObject::SetFreeze(bool freeze)
{
}

void CharacterObject::SetRenderModel(DirectX::Scene::IModelNode * pMesh, int LoD)
{
	m_pSkinModel = dynamic_cast<ISkinningModel*>(pMesh);
	m_BoneTransforms.resize(m_pSkinModel->GetBonesCount());
	//XMMATRIX identity = XMMatrixIdentity();
	//for (auto& t : m_BoneTransforms)
	//	XMStoreA(t, identity);

	if (m_pSkinModel == nullptr && pMesh != nullptr)
	{
		throw std::exception("Render model doesn't support Skinning interface.");
	}

	VisualObject::SetRenderModel(pMesh, LoD);
}

void CharacterObject::Update(time_seconds const & time_delta)
{
	SceneObject::Update(time_delta);
	if (m_pCurrentAction != nullptr)
	{
		m_CurrentActionTime += time_delta;
		this->MapCurrentFrameForUpdate();
		m_pCurrentAction->GetFrameAt(m_CurrentFrame, m_CurrentActionTime);
		//ScaleFrame(m_CurrentFrame, Armature().default_frame(), 0.95);
		//m_CurrentFrame.RebuildGlobal(Armature());
		this->ReleaseCurrentFrameFrorUpdate();
	}

	if (m_IsAutoDisplacement)
	{
		ComputeVelocityFrame(time_delta);
		DisplaceByVelocityFrame();
	}

	if (m_pSkinModel)
	{
		auto pBones = m_BoneTransforms.data();
		FrameTransformMatrix(pBones, Armature().default_frame(), m_CurrentFrame, m_pSkinModel->GetBonesCount());
	}
}

RenderFlags CharacterObject::GetRenderFlags() const
{
	if (m_pSkinModel)
		return RenderFlags::Skinable | RenderFlags::OpaqueObjects;
	return RenderFlags::OpaqueObjects;
}

bool CharacterObject::IsVisible(const BoundingGeometry & viewFrustum) const
{
	return VisualObject::IsVisible(viewFrustum);
}

void CharacterObject::Render(IRenderContext * pContext, DirectX::IEffect* pEffect)
{
	if (g_ShowCharacterMesh)
	{
		auto pSkinEffect = dynamic_cast<IEffectSkinning*>(pEffect);

		if (pSkinEffect)
			pSkinEffect->SetBoneTransforms(GetBoneTransforms(), m_BoneTransforms.size());

		VisualObject::Render(pContext, pEffect);
	}

	auto pLS = dynamic_cast<IEffectLightsShadow*>(pEffect);
	if (g_DebugView && pLS != nullptr)
	{
		using namespace DirectX;
		using Visualizers::g_PrimitiveDrawer;
		const auto& frame = m_CurrentFrame;
		//g_PrimitiveDrawer.Begin();
		//const auto& dframe = Armature().default_frame();

		auto trans = this->GlobalTransformMatrix();

		// Draw X-ray armature
		if (g_DrawXRayArmature)
		{
			XMVECTOR color = Colors::Yellow.v;
			color = DirectX::XMVectorSetW(color, Opticity());

			ID3D11DepthStencilState *pDSS = NULL;
			UINT StencilRef;
			pContext->OMGetDepthStencilState(&pDSS, &StencilRef);
			pContext->OMSetDepthStencilState(g_PrimitiveDrawer.GetStates()->DepthNone(), StencilRef);
			DrawArmature(this->Armature(), frame, color, trans, g_DebugArmatureThinkness / this->GetGlobalTransform().Scale.x);
			pContext->OMSetDepthStencilState(pDSS, StencilRef);
		}

		// Draw bone bounding boxes
		if (g_DrawPartBoxes)
		{
			auto& drawer = g_PrimitiveDrawer;
			drawer.SetWorld(trans);
			drawer.Begin();
			if (g_ShowCharacterMesh && m_pSkinModel)
			{
				auto boxes = m_pSkinModel->GetBoneBoundingBoxes();
				for (int i = 0; i < m_pArmature->size(); i++)
				{
					BoundingGeometry geo(boxes[i]);
					geo.Transform(geo, XMLoadA(m_BoneTransforms[i]));
					DrawGeometryOutline(geo, Colors::Orange);
				}
			}
			drawer.End();
		}
	}
}

void XM_CALLCONV CharacterObject::UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection)
{
	if (g_DebugView)
	{
		using namespace DirectX;
		using Visualizers::g_PrimitiveDrawer;
		g_PrimitiveDrawer.SetView(view);
		g_PrimitiveDrawer.SetProjection(projection);
	}
	VisualObject::UpdateViewMatrix(view, projection);
}


CharacterObject::CharacterObject()
{
	m_IsAutoDisplacement = false;
	m_SpeedFilter.SetCutoffFrequency(60);
	m_SpeedFilter.SetUpdateFrequency(&updateFrequency);
}


CharacterObject::~CharacterObject()
{
}

void CharacterObject::Parse(const ParamArchive * store)
{
	VisualObject::Parse(store);

	auto& m_assets = Scene->Assets();
	const char* behv_src = nullptr;
	GetParam(store, "behavier", behv_src);
	if (behv_src != nullptr && strlen(behv_src) != 0)
	{
		if (behv_src[0] == '{') // asset reference
		{
			const std::string key(behv_src + 1, behv_src + strlen(behv_src) - 1);
			SetBehavier(*m_assets.GetBehavier(key));
		}
	}
	else
	{
		auto inlineBehave = GetFirstChildArchive(store, "creature.behavier");
		if (inlineBehave)
		{
			inlineBehave = GetFirstChildArchive(inlineBehave);
			auto behavier = m_assets.ParseBehavier(inlineBehave);
			SetBehavier(*behavier);
		}
	}

	const char* action = nullptr;
	GetParam(store, "action", action);
	if (action)
		StartAction(action);
}

void CharacterObject::EnabeAutoDisplacement(bool is_enable)
{
	m_IsAutoDisplacement = is_enable;
	m_SpeedFilter.Reset();
}

void Causality::DrawArmature(const IArmature & armature, ArmatureFrameConstView frame, const Color* colors, const Matrix4x4 & world, float thinkness)
{
	using DirectX::Visualizers::g_PrimitiveDrawer;

	// Invaliad frame
	if (frame.size() < armature.size())
		return;

	g_PrimitiveDrawer.SetWorld(world);
	//g_PrimitiveDrawer.Begin();
	for (auto& joint : armature.joints())
	{
		auto& bone = frame[joint.ID];
		XMVECTOR ep = bone.GblTranslation;

		if (!joint.is_root())
		{
			auto& pbone = frame[joint.parent()->ID];
			XMVECTOR sp = pbone.GblTranslation;

			//g_PrimitiveDrawer.DrawLine(sp, ep, color);
			g_PrimitiveDrawer.DrawCylinder(sp, ep, thinkness, colors[joint.ID]);
		}
		g_PrimitiveDrawer.DrawSphere(ep, thinkness * 1.5f, colors[joint.ID]);
	}
	//g_PrimitiveDrawer.End();
}


void Causality::DrawArmature(const IArmature & armature, ArmatureFrameConstView frame, const Color & color, const Matrix4x4 & world, float thinkness)
{
	using DirectX::Visualizers::g_PrimitiveDrawer;

	// Invaliad frame
	if (frame.size() < armature.size())
		return;

	g_PrimitiveDrawer.SetWorld(world);
	//g_PrimitiveDrawer.Begin();
	for (auto& joint : armature.joints())
	{
		auto& bone = frame[joint.ID];
		XMVECTOR ep = bone.GblTranslation;

		if (!joint.is_root())
		{
			auto& pbone = frame[joint.parent()->ID];
			XMVECTOR sp = pbone.GblTranslation;

			//g_PrimitiveDrawer.DrawLine(sp, ep, color);
			g_PrimitiveDrawer.DrawCylinder(sp, ep, thinkness, color);
		}
		g_PrimitiveDrawer.DrawSphere(ep, thinkness * 1.5f, color);
	}
	//g_PrimitiveDrawer.End();
}

void CharacterGlowParts::Render(IRenderContext * pContext, DirectX::IEffect * pEffect)
{
	auto pSGEffect = dynamic_cast<ShadowMapGenerationEffect*> (pEffect);
	if (pSGEffect && pSGEffect->GetShadowFillMode() == ShadowMapGenerationEffect::BoneColorFill)
	{
		pSGEffect->SetBoneColors(reinterpret_cast<XMVECTOR*>(m_BoneColors.data()), m_BoneColors.size());
		pSGEffect->SetBoneTransforms(m_pCharacter->GetBoneTransforms(), m_BoneColors.size());
		auto pModel = m_pCharacter->RenderModel();
		if (pModel)
		{
			pModel->Render(pContext, m_pCharacter->GlobalTransformMatrix(), pEffect); // Render parent model with customized effect
		}
	}
}

RenderFlags CharacterGlowParts::GetRenderFlags() const
{
	return RenderFlags::BloomEffectSource | RenderFlags::Skinable;
}

void CharacterGlowParts::ResetBoneColor(const DirectX::Color & color)
{
	std::fill(m_BoneColors.begin(), m_BoneColors.end(), color);
}

void CharacterGlowParts::OnParentChanged(SceneObject* oldParent)
{
	Initialize();
}

void CharacterGlowParts::Initialize()
{
	m_pCharacter = this->FirstAncesterOfType<CharacterObject>();
	m_BoneColors.resize(m_pCharacter->Armature().size());
	for (auto& color : m_BoneColors)
	{
		color.A(0);
	}
}
