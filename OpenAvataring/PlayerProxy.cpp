#include "pch.h"
#include <PrimitiveVisualizer.h>
#include <fstream>
#include <algorithm>
#include <ppl.h>
#include <filesystem>
#include <Models.h>
//#include <boost\filesystem.hpp>
//#include <random>
//#include <unsupported\Eigen\fft>
//#pragma warning (disable:4554)
//#include <unsupported\Eigen\CXX11\Tensor>

#include "GaussianProcess.h"
//#include "QudraticAssignment.h"

#include "ArmatureParts.h"
#include "ClipMetric.h"

#include "PlayerProxy.h"
#include "ArmatureTransforms.h"
#include "Cca.h"
#include "EigenExtension.h"

#include "Causality\Scene.h"
#include "Causality\Settings.h"
#include "Causality\CameraObject.h"



//					When this flag set to true, a CCA will be use to find the general linear transform between Player 'Limb' and Character 'Limb'

//float				g_NoiseInterpolation = 1.0f;


using namespace Causality;
using namespace Eigen;
using namespace std;
using namespace ArmaturePartFeatures;

REGISTER_SCENE_OBJECT_IN_PARSER(player_controller, PlayerProxy);
REGISTER_SCENE_OBJECT_IN_PARSER(kinect_visualizer, KinectVisualizer);

//using boost::filesystem::path;
using experimental::filesystem::path;

path g_LogRootDir = "Log";
static const char*  DefaultAnimationSet = "walk";
Eigen::RowVector3d	g_NoiseInterpolation = { 1.0,1.0,1.0 };
const static Eigen::IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

std::map<string, string> g_DebugLocalMotionAction;
bool					 g_DebugLocalMotion = false;

static const DirectX::XMVECTORF32 HumanBoneColors[JointType_Count] = {
	{ 0.0f,0.0f,0.0f,0.0f },//JointType_SpineBase = 0,
	{ 0.4f,0.4f,0.1f,0.0f },//JointType_SpineMid = 1,
	{ 0.9f,0.3f,0.9f,1.0f },//JointType_Neck = 2,
	{ 0.9f,0.3f,0.9f,1.0f },//JointType_Head = 3,
	{ 0.9f,0.3f,0.3f,1.0f },//JointType_ShoulderLeft = 4,
	{ 0.9f,0.3f,0.3f,1.0f },//JointType_ElbowLeft = 5,
	{ 0.9f,0.3f,0.3f,1.0f },//JointType_WristLeft = 6,
	{ 0.9f,0.3f,0.3f,1.0f },//JointType_HandLeft = 7,
	{ 0.3f,0.3f,0.9f,1.0f },//JointType_ShoulderRight = 8,
	{ 0.3f,0.3f,0.9f,1.0f },//JointType_ElbowRight = 9,
	{ 0.3f,0.3f,0.9f,1.0f },//JointType_WristRight = 10,
	{ 0.3f,0.3f,0.9f,1.0f },//JointType_HandRight = 11,
	{ 0.9f,0.9f,0.3f,1.0f },//JointType_HipLeft = 12,
	{ 0.9f,0.9f,0.3f,1.0f },//JointType_KneeLeft = 13,
	{ 0.9f,0.9f,0.3f,1.0f },//JointType_AnkleLeft = 14,
	{ 0.9f,0.9f,0.3f,1.0f },//JointType_FootLeft = 15,
	{ 0.3f,0.9f,0.9f,1.0f },//JointType_HipRight = 16,
	{ 0.3f,0.9f,0.9f,1.0f },//JointType_KneeRight = 17,
	{ 0.3f,0.9f,0.9f,1.0f },//JointType_AnkleRight = 18,
	{ 0.3f,0.9f,0.9f,1.0f },//JointType_FootRight = 19,
	{ 0.0f,0.0f,0.0f,0.0f },//JointType_SpineShoulder = 20,
	{ 0.9f,0.3f,0.3f,1.0f },//JointType_HandTipLeft = 21,
	{ 0.9f,0.3f,0.3f,1.0f },//JointType_ThumbLeft = 22,
	{ 0.3f,0.3f,0.9f,1.0f },//JointType_HandTipRight = 23,
	{ 0.3f,0.3f,0.9f,1.0f },//JointType_ThumbRight = 24,
	//JointType_Count = (JointType_ThumbRight + 1)
};

/*
pair<JointType, JointType> XFeaturePairs[] = {
	{ JointType_SpineBase, JointType_SpineShoulder },
	{ JointType_SpineShoulder, JointType_Head },
	{ JointType_ShoulderLeft, JointType_ElbowLeft },
	{ JointType_ShoulderRight, JointType_ElbowRight },
	{ JointType_ElbowLeft, JointType_HandLeft },
	{ JointType_ElbowRight, JointType_HandRight },
	{ JointType_HipLeft, JointType_KneeLeft },
	{ JointType_HipRight, JointType_KneeRight },
	{ JointType_KneeLeft, JointType_AnkleLeft },
	{ JointType_KneeRight, JointType_AnkleRight },
	//{ JointType_HandLeft, JointType_HandTipLeft },
	//{ JointType_HandRight, JointType_HandTipRight },
	//{ JointType_HandLeft, JointType_ThumbLeft },
	//{ JointType_HandRight, JointType_ThumbRight },
};

JointType KeyJoints[] = {
	JointType_SpineBase,		//1
	JointType_SpineShoulder,	//2
	JointType_Head,				//3
	JointType_ShoulderLeft,		//4
	JointType_ElbowLeft,		//5
	JointType_WristLeft,		//6
	JointType_ShoulderRight,	//7
	JointType_ElbowRight,		//8
	JointType_WristRight,		//9
	JointType_HipLeft,			//10
	JointType_KneeLeft,			//11
	JointType_AnkleLeft,		//12
	JointType_HipRight,			//13
	JointType_KneeRight,		//14
	JointType_AnkleRight,		//15
};
*/
float BoneRadius[JointType_Count] = {
};

#define BEGIN_TO_END(range) range.begin(), range.end()


void SetGlowBoneColor(CharacterGlowParts* glow, const Causality::ShrinkedArmature & sparts, const CharacterController& controller);

// Player Proxy methods
void PlayerProxy::StreamPlayerFrame(const IArmatureStreamAnimation& body, const IArmatureStreamAnimation::FrameType& frame)
{
	using namespace Eigen;
	using namespace DirectX;

	m_pushFrame = frame;
	m_newFrameAvaiable = true;

	if (g_IngnoreInputRootRotation)
	{
		RemoveFrameRootTransform(m_pushFrame, *m_pPlayerArmature);
	}

	bool newMetric = m_CyclicInfo.StreamFrame(m_pushFrame);
	if (newMetric && !m_mapTaskOnGoing/* && (m_mapTask.empty() || m_mapTask.is_done())*/)
	{
		m_mapTaskOnGoing = true;
		m_mapTask = concurrency::create_task([this]() {
			auto idx = MapCharacterByLatestMotion();
			m_mapTaskOnGoing = false;
		});
	}

}

void PlayerProxy::ResetPlayer(IArmatureStreamAnimation * pOld, IArmatureStreamAnimation * pNew)
{
	StopUpdateThread();
	SetActiveController(-1);

	if (!pOld || (pNew && &pNew->GetArmature() != &pOld->GetArmature()))
	{
		ResetPlayerArmature(&pNew->GetArmature());
	}

	m_CyclicInfo.ResetStream();
	m_CyclicInfo.EnableCyclicMotionDetection(true);
	m_updateTime = 0;
	m_updateCounter = 0;

	if (pNew)
		StartUpdateThread();
}

void PlayerProxy::ResetPlayerArmature(const IArmature* playerArmature)
{
	m_pPlayerArmature = playerArmature;
	InitializeShrinkedPlayerArmature();
	cout << "Initializing Cyclic Info..." << endl;
	m_CyclicInfo.Initialize(*m_pParts, time_seconds(0.5), time_seconds(3), 30, 0);
	cout << "Cyclic Info Initited!" << endl;
}

inline std::ostream& operator<<(std::ostream& os, const Joint& joint)
{
	os << joint.Name;
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const Joint* joint)
{
	os << joint->Name;
	return os;
}

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T> &vec)
{
	cout << '{';
	for (auto& t : vec)
	{
		cout << t << ", ";
	}
	cout << "\b\b}";
	return os;
}



PlayerProxy::PlayerProxy()
	: m_IsInitialized(false),
	m_pSelector(nullptr),
	m_CurrentIdx(-1),
	current_time(0),
	m_mapTaskOnGoing(false),
	m_EnableOverShoulderCam(true),
	m_DefaultCameraFlag(true),
	m_updateCounter(0),
	m_updateTime(0),
	m_pParts(new ShrinkedArmature()),
	m_pPlayerArmature(nullptr)
{
	//ResetPlayerArmature(TrackedBody::BodyArmature.get());
	Register();
	m_stopUpdate = true;
	m_IsInitialized = true;
}

void PlayerProxy::InitializeShrinkedPlayerArmature()
{
	m_pParts->SetArmature(*m_pPlayerArmature);

	{
		cout << "Player Armature Parts" << endl;
		int i = 0;
		auto& parts = *m_pParts;
		for (auto pPart : parts)
		{
			auto& part = *pPart;
			cout << "Part[" << i++ << "] = " << part.Joints;
			if (part.SymetricPair != nullptr)
			{
				cout << " <--> {" << part.SymetricPair->Joints[0] << "...}";
			}
			cout << endl;
		}
	}

	cout << "Armature Proportions : " << endl;
	for (auto pPart : *m_pParts)
	{
		auto& part = *pPart;
		cout << "Part " << part.Joints << " = ";
		cout << part.ChainLength << '|' << part.LengthToRoot << endl;
	}
}

void PlayerProxy::StartUpdateThread()
{
	if (!m_stopUpdate) return;
	m_stopUpdate = false;
	m_updateCounter = 0;
	m_lastUpdateTime = chrono::system_clock::now();
	m_updateThread = thread(std::bind(&PlayerProxy::UpdateThreadRuntime, this));
}

void PlayerProxy::StopUpdateThread()
{
	m_stopUpdate = true;
	if (m_updateThread.joinable())
	{
		m_updateThread.join();
	}
}


PlayerProxy::~PlayerProxy()
{
	StopUpdateThread();
	Unregister();
	//std::ofstream fout("handpos.txt", std::ofstream::out);

	//fout.close();
}

void PlayerProxy::AddChild(SceneObject* pChild)
{
	SceneObject::AddChild(pChild);
	auto pChara = dynamic_cast<CharacterObject*>(pChild);
	if (pChara)
	{
		auto settings = Scene->GetSceneSettings();

		m_Controllers.emplace_back();
		auto& controller = m_Controllers.back();
		controller.ID = m_Controllers.size() - 1;
		controller.Initialize(*pChara, settings);
		pChara->SetOpticity(1.0f);

		if (g_DebugLocalMotion)
		{
			g_DebugLocalMotionAction[pChara->Name] = pChara->CurrentActionName();
			pChara->StopAction();
		}

		auto glow = pChara->FirstChildOfType<CharacterGlowParts>();
		if (glow == nullptr)
		{
			glow = new CharacterGlowParts();
			glow->Scene = this->Scene;
			glow->SetEnabled(false);
			pChara->AddChild(glow);
		}
	}
}

void PlayerProxy::Parse(const ParamArchive * store)
{
	auto sel = GetFirstChildArchive(store, "player_controller.selector");
	sel = GetFirstChildArchive(sel);
	string name = GetArchiveName(sel);
	if (name == "kinect_player_selector")
	{
		auto pKinect = Devices::KinectSensor::GetForCurrentView();
		KinectPlayerSelector::SelectionMode mode;
		unsigned umode = KinectPlayerSelector::ClosestStickly;
		GetParam(sel, "mode", umode);
		mode = (KinectPlayerSelector::SelectionMode)umode;
		auto pSelector = make_shared<KinectPlayerSelector>(pKinect.get(), mode);
		SetPlayerSelector(pSelector);
		pKinect->Start();
	}
	else if (name == "leap_selector")
	{

	}
	else if (name == "virutal_character_selector")
	{

	}
}

void SetGlowBoneColorPartPair(Causality::CharacterGlowParts * glow, int Jx, int Jy, const DirectX::XMVECTORF32 *colors, const Causality::ShrinkedArmature & sparts, const Causality::ShrinkedArmature & cparts);

void SetGlowBoneColor(CharacterGlowParts* glow, const Causality::ShrinkedArmature & sparts, const CharacterController& controller)
{
	auto pTrans = &controller.Binding();
	auto pCcaTrans = dynamic_cast<const BlockizedCcaArmatureTransform*>(pTrans);
	auto pPartTrans = dynamic_cast<const PartilizedTransformer*>(pTrans);

	auto& cparts = controller.ArmatureParts();
	auto& colors = HumanBoneColors;

	//auto& carmature = controller.Character().Armature();
	//for (int i = 0; i <= carmature.size(); i++)
	//{
	//	glow->SetBoneColor(i, DirectX::Colors::Orange.v);
	//}
	//return;

	glow->ResetBoneColor(Math::Colors::Transparent.v);

	if (pCcaTrans)
	{
		for (auto& tp : pCcaTrans->Maps)
		{
			auto Jx = tp.Jx, Jy = tp.Jy;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, sparts, cparts);
		}
	}
	else if (pPartTrans)
	{
		for (auto& tp : pPartTrans->ActiveParts)
		{
			auto Jx = tp.SrcIdx, Jy = tp.DstIdx;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, sparts, cparts);
		}
		for (auto& tp : pPartTrans->DrivenParts)
		{
			auto Jx = tp.SrcIdx, Jy = tp.DstIdx;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, sparts, cparts);
		}
		for (auto& tp : pPartTrans->AccesseryParts)
		{
			auto Jx = tp.SrcIdx, Jy = tp.DstIdx;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, sparts, cparts);
		}
	}

}

void SetGlowBoneColorPartPair(Causality::CharacterGlowParts * glow, int Jx, int Jy, const DirectX::XMVECTORF32 *colors, const Causality::ShrinkedArmature & sparts, const Causality::ShrinkedArmature & cparts)
{
	using namespace Math;
	XMVECTOR color;
	if (Jx == NoInputParts)
		color = Colors::Transparent;
	else if (Jx == ActiveAndDrivenParts)
		color = Colors::LightBlue;
	else if (Jx == ActiveParts)
		color = Colors::ForestGreen;
	else if (Jx >= 0)
		color = colors[sparts[Jx]->Joints.front()->ID];

	using namespace DirectX;
	color = XMVectorSetW(color, 0.5f);
	for (auto joint : cparts[Jy]->Joints)
	{
		glow->SetBoneColor(joint->ID, color);
	}
}

void PlayerProxy::SetActiveController(int idx)
{
	std::lock_guard<std::mutex> guard(m_controlMutex);

	if (idx >= 0)
		idx = idx % m_Controllers.size();

	if (idx == -1)
		ResetPrimaryCameraPoseToDefault();

	for (auto& c : m_Controllers)
	{
		if (c.ID != idx)
		{
			auto& chara = c.Character();

			if (!g_DebugLocalMotion && !g_DebugLocalMotionAction[chara.Name].empty())
			{
				auto& action = g_DebugLocalMotionAction[chara.Name];
				chara.StartAction(action);
				g_DebugLocalMotionAction[chara.Name] = "";
			}

			chara.SetOpticity(0.5f);
			auto glow = chara.FirstChildOfType<CharacterGlowParts>();
			glow->SetEnabled(false);

			if (c.ID == m_CurrentIdx && m_CurrentIdx != idx)
			{
				chara.SetPosition(c.CMapRefPos);
				chara.SetOrientation(c.CMapRefRot);
				chara.EnabeAutoDisplacement(false);
			}
		}
	}

	if (m_CurrentIdx != idx)
	{
		m_CurrentIdx = idx;
		if (m_CurrentIdx != -1)
		{
			auto& controller = GetController(m_CurrentIdx);
			auto& chara = controller.Character();

			if (!g_DebugLocalMotion && !chara.CurrentActionName().empty())
			{
				g_DebugLocalMotionAction[chara.Name] = chara.CurrentActionName();
				chara.StopAction();
			}

			chara.SetOpticity(1.0f);

			auto glow = chara.FirstChildOfType<CharacterGlowParts>();
			if (glow)
			{
				glow->SetEnabled(true);
				std::lock_guard<std::mutex> guard(controller.GetBindingMutex());
				SetGlowBoneColor(glow, *m_pParts, controller);
			}

			assert(m_pSelector && m_pSelector->Get());
			auto &player = *m_pSelector->Get();
			auto& frame = player.PeekFrame();
			auto pose = frame[m_pPlayerArmature->root()->ID];
			controller.SetReferenceSourcePose(pose);

			chara.EnabeAutoDisplacement(g_UsePersudoPhysicsWalk);
		}
	}
	else
	{
		if (m_CurrentIdx == -1)
			return;

		auto& controller = GetController(m_CurrentIdx);
		auto& chara = controller.Character();
		controller.SetBinding(nullptr);

		auto glow = chara.FirstChildOfType<CharacterGlowParts>();
		if (glow)
		{
			SetGlowBoneColor(glow, *m_pParts, controller);
			glow->SetEnabled(!g_DebugView);
		}
	}

	//if (m_CurrentIdx >= 0)
	//	StartUpdateThread();
	//else
	//	StopUpdateThread();
}

int PlayerProxy::MapCharacterByLatestMotion()
{
	if (!m_pSelector || !m_pSelector->Get())
		return -1;

	auto& player = *m_pSelector->Get();

	CharacterController* pControl = nullptr;
	{
		std::lock_guard<std::mutex> guard(m_CyclicInfo.AqucireFacadeMutex());
		cout << "FacadeLock Aquired" << endl;

		//std::this_thread::sleep_for(std::chrono::seconds(1));

		for (auto& controller : m_Controllers)			//? <= 5 character
		{
			if (!controller.IsReady)
				continue;
			controller.CreateControlBinding(m_CyclicInfo.AsFacade());

			if (!pControl || controller.CharacterScore > pControl->CharacterScore)
				pControl = &controller;
		}

		if (pControl)
			// Disable re-matching when the controller has not request
			m_CyclicInfo.EnableCyclicMotionDetection(false);

		cout << "FacadeLock Releasing" << endl;
	}

	if (!pControl) return -1;

	SetActiveController(pControl->ID);

	return pControl->ID;
}




bool PlayerProxy::IsMapped() const { return m_CurrentIdx >= 0; }

const CharacterController & PlayerProxy::CurrentController() const {
	for (auto& c : m_Controllers)
	{
		if (c.ID == m_CurrentIdx)
			return c;
	}
}

CharacterController & PlayerProxy::CurrentController() {
	for (auto& c : m_Controllers)
	{
		if (c.ID == m_CurrentIdx)
			return c;
	}
	throw;
}

const CharacterController & PlayerProxy::GetController(int state) const {
	for (auto& c : m_Controllers)
	{
		if (c.ID == state)
			return c;
	}
	throw;
}

CharacterController & PlayerProxy::GetController(int state)
{
	for (auto& c : m_Controllers)
	{
		if (c.ID == state)
			return c;
	}
	throw;
}

void PlayerProxy::OnKeyUp(const KeyboardEventArgs & e)
{
	if (e.Key == VK_OEM_PERIOD || e.Key == '.' || e.Key == '>')
	{
		int idx = (m_CurrentIdx + 1) % m_Controllers.size();
		SetActiveController(idx);
	}
	else if (e.Key == VK_OEM_COMMA || e.Key == ',' || e.Key == '<')
	{
		if (m_Controllers.size() > 0)
		{
			int idx = m_CurrentIdx - 1;
			if (idx < 0)
				idx = m_Controllers.size() - 1;
			SetActiveController(idx);
		}
	}
	else if (e.Key == 'L')
	{
		// this behavier should not change in mapped mode
		if (IsMapped()) return;

		g_DebugLocalMotion = !g_DebugLocalMotion;
		if (g_DebugLocalMotion)
		{
			for (auto& controller : m_Controllers)
			{
				auto& chara = controller.Character();
				g_DebugLocalMotionAction[chara.Name] = chara.CurrentActionName();
				chara.StopAction();
			}
		}
		else
		{
			for (auto& controller : m_Controllers)
			{
				auto& chara = controller.Character();
				auto& action = g_DebugLocalMotionAction[chara.Name];
				if (!action.empty())
					chara.StartAction(action);
				g_DebugLocalMotionAction[chara.Name] = "";
			}
		}
	}
	else if (e.Key == VK_UP || e.Key == VK_DOWN)
	{
		for (auto& controller : m_Controllers)
		{
			auto& chara = controller.Character();
			auto& clips = chara.Behavier().Clips();
			auto& idx = controller.CurrentActionIndex;
			if (e.Key == VK_UP)
				idx = (idx + 1) % clips.size();
			else
				idx = idx == 0 ? clips.size() - 1 : idx - 1;

			if (g_DebugLocalMotion)
				g_DebugLocalMotionAction[chara.Name] = clips[idx].Name;
			else
				chara.StartAction(clips[idx].Name);
		}
	}
	else if (e.Key == 'P')
	{
		g_EnableDependentControl = !g_EnableDependentControl;
		cout << "Enable Dependency Control = " << g_EnableDependentControl << endl;
	}
	else if (e.Key == 'C')
	{
		m_EnableOverShoulderCam = !m_EnableOverShoulderCam;
		//g_UsePersudoPhysicsWalk = m_EnableOverShoulderCam;
		cout << "Over Shoulder Camera Mode = " << m_EnableOverShoulderCam << endl;
		//cout << "Persudo-Physics Walk = " << g_UsePersudoPhysicsWalk << endl;
	}
	else if (e.Key == 'V')
	{
		g_UsePersudoPhysicsWalk = !g_UsePersudoPhysicsWalk;
		cout << "Persudo-Physics Walk = " << g_UsePersudoPhysicsWalk << endl;
		for (auto& controller : m_Controllers)
		{
			controller.Character().EnabeAutoDisplacement(g_UsePersudoPhysicsWalk && controller.ID == m_CurrentIdx);
		}
	}
	//else if (e.Key == 'M')
	//{
	//	g_MirrowInputX = !g_MirrowInputX;
	//	cout << "Kinect Input Mirrowing = " << g_MirrowInputX << endl;
	//	m_pKinect->EnableMirrowing(g_MirrowInputX);
	//}
	else if (e.Key == VK_BACK)
	{
		m_CyclicInfo.ResetStream();
		m_DefaultCameraFlag = true;
	}
	else if (e.Key == VK_NUMPAD1)
	{
		g_NoiseInterpolation[0] -= 0.1f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD3)
	{
		g_NoiseInterpolation[0] += 0.1f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD2)
	{
		g_NoiseInterpolation[0] = 1.0f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD4)
	{
		g_NoiseInterpolation[1] -= 0.1f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD6)
	{
		g_NoiseInterpolation[1] += 0.1f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD5)
	{
		g_NoiseInterpolation[1] = 1.0f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD7)
	{
		g_NoiseInterpolation[2] -= 0.1f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD9)
	{
		g_NoiseInterpolation[2] += 0.1f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == VK_NUMPAD8)
	{
		g_NoiseInterpolation[2] = 1.0f;
		cout << "Local Motion Sythesis Jaming = " << g_NoiseInterpolation << endl;
	}
	else if (e.Key == 'R')
	{
		ResetPrimaryCameraPoseToDefault();

	}
}

void PlayerProxy::OnKeyDown(const KeyboardEventArgs & e)
{
}

RenderFlags Causality::PlayerProxy::GetRenderFlags() const
{
	return RenderFlags::SpecialEffects;
}

void AddNoise(ArmatureFrameView frame, float sigma)
{
	//static std::random_device rd;
	//static std::mt19937 gen(rd());

	//std::normal_distribution<float> nd(1.0f, sigma);

	//for (auto& bone : frame)
	//{
	//	bone.GblTranslation *= 0.95;//nd(gen);
	//}
}

void PlayerProxy::UpdateThreadRuntime()
{
	while (!(bool)m_stopUpdate)
	{
		if (!(bool)m_newFrameAvaiable) continue;

		auto& player = *m_pSelector->Get();

		if (!m_pSelector->Get())
		{
			cout << "Player Selector Lost, stop update." << endl;
			return;
		}

		if (!player.IsAvailable() || !player.ReadLatestFrame())
			continue;

		// Update time / frame
		auto now = std::chrono::system_clock::now();
		time_seconds dts = now - m_lastUpdateTime;
		double dt = dts.count();
		m_lastUpdateTime = now;

		m_lastFrame = m_currentFrame;
		m_currentFrame = player.PeekFrame();

		m_newFrameAvaiable = false;

		// we need lastFrame and currentFrame be both valiad, thus 2 frame
		if (++m_updateCounter < 2)
			continue;

		const auto& frame = m_currentFrame;
		const auto& lastFrame = m_lastFrame;

		g_RevampLikilyhoodThreshold = 0.5;
		g_RevampLikilyhoodTimeThreshold = 1.0;

		if (g_ForceRemappingAlwaysOn)
			m_CyclicInfo.EnableCyclicMotionDetection();

		if (IsMapped() && m_controlMutex.try_lock())
		{
			std::lock_guard<std::mutex> guard(m_controlMutex,std::adopt_lock);
			cout << "getting mutext" << endl;
			auto& controller = CurrentController();
			float lik = controller.UpdateTargetCharacter(frame, lastFrame, dt);

			// Check if we need to "Revamp" Control Binding
			if (lik < g_RevampLikilyhoodThreshold)
			{
				m_LowLikilyTime += dt;
				if (m_LowLikilyTime > g_RevampLikilyhoodTimeThreshold)
				{
					m_CyclicInfo.EnableCyclicMotionDetection();
				}
			}
			else
			{
				m_CyclicInfo.EnableCyclicMotionDetection(false);
				m_LowLikilyTime = 0;
			}
		}
		else
		{

		}
	}
}

void PlayerProxy::Update(time_seconds const & time_delta)
{
	SceneObject::Update(time_delta);
	using namespace std;
	using namespace Eigen;

	if (!m_IsInitialized)
		return;

	if (g_DebugLocalMotion && !IsMapped())
	{
		UpdateSelfMotionBinder(time_delta);
		return;
	}


	if (IsMapped() && m_EnableOverShoulderCam && m_pSelector->Get() && m_pSelector->Get()->IsAvailable())
		UpdatePrimaryCameraForTrack();

	// no new frame is coming
	static long long frame_count = 0;
}

void PlayerProxy::UpdateSelfMotionBinder(const Causality::time_seconds & time_delta)
{
	current_time += time_delta;
	ArmatureFrame last_frame;
	ArmatureFrame anotherFrame, anotherLastFrame;
	for (auto& controller : m_Controllers)
	{
		if (!controller.IsReady)
			continue;
		auto& chara = controller.Character();
		auto& actionName = g_DebugLocalMotionAction[chara.Name];
		if (actionName.empty())
			continue;
		auto& action = controller.Character().Behavier()[actionName];
		auto& target_frame = controller.Character().MapCurrentFrameForUpdate();
		auto frame = controller.Character().Armature().default_frame();

		target_frame = frame;
		last_frame = frame;
		action.GetFrameAt(frame, current_time);
		action.GetFrameAt(last_frame, current_time - time_delta);

		//auto& anotheraction = controller.Character().Behavier()["run"];
		//anotheraction.GetFrameAt(anotherFrame, current_time);
		//anotheraction.GetFrameAt(anotherLastFrame, current_time - time_delta);

		//for (size_t i = 0; i < frame.size(); i++)
		//{
		//	frame[i].GblTranslation = DirectX::XMVectorLerp(frame[i].GblTranslation, anotherFrame[i].GblTranslation, g_NoiseInterpolation);
		//	last_frame[i].GblTranslation = DirectX::XMVectorLerp(last_frame[i].GblTranslation, anotherLastFrame[i].GblTranslation, g_NoiseInterpolation);
		//}

		// Add motion to non-active joints that visualize more about errors for active joints
		target_frame = frame;
		//AddNoise(frame, .1f);
		controller.SelfBinding().Transform(target_frame, frame, last_frame, time_delta.count());
	}
}

void PlayerProxy::UpdatePrimaryCameraForTrack()
{
	auto& camera = *this->Scene->PrimaryCamera();
	auto& cameraPos = dynamic_cast<SceneObject&>(camera);
	auto& contrl = this->CurrentController();
	auto& chara = contrl.Character();
	using namespace DirectX;
	XMVECTOR ext = XMLoad(chara.RenderModel()->GetBoundingBox().Extents);
	ext = XMVector3LengthEst(ext);
	ext *= chara.GetGlobalTransform().Scale;

	if (m_DefaultCameraFlag)
	{
		m_DefaultCameraFlag = false;
		m_DefaultCameraPose.Translation = cameraPos.GetPosition();
		m_DefaultCameraPose.Rotation = cameraPos.GetOrientation();
	}

	cameraPos.SetPosition((XMVECTOR)chara.GetPosition() + XMVector3Rotate(XMVectorMultiplyAdd(ext, XMVectorSet(-2.0f, 2.0f, -2.0f, 0.0f), XMVectorSet(-0.5f, 0.5, -0.5, 0)), chara.GetOrientation()));
	camera.GetView()->FocusAt((XMVECTOR)chara.GetPosition() + XMVector3Rotate(XMVectorMultiplyAdd(ext, XMVectorSet(-2.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(-0.5f, 0.5, -0.5, 0)), chara.GetOrientation()), g_XMIdentityR1.v);
}

void PlayerProxy::ResetPrimaryCameraPoseToDefault()
{
	// Camera pose not changed by Over Shoulder view
	if (m_DefaultCameraFlag)
		return;

	auto& camera = *this->Scene->PrimaryCamera();
	auto& cameraPos = dynamic_cast<SceneObject&>(camera);

	m_DefaultCameraFlag = true;
	cameraPos.SetPosition(m_DefaultCameraPose.Translation);
	cameraPos.SetOrientation(m_DefaultCameraPose.Rotation);
}

bool PlayerProxy::IsVisible(const DirectX::BoundingGeometry & viewFrustum) const
{
	return true;
}

void DrawJammedGuidingVectors(const ShrinkedArmature & barmature, ArmatureFrameConstView frame, const Color & color, const Matrix4x4 & world, float thinkness = 0.015f)
{
	using DirectX::Visualizers::g_PrimitiveDrawer;
	using namespace DirectX;
	if (frame.size() == 0)
		return;
	//g_PrimitiveDrawer.SetWorld(world);
	g_PrimitiveDrawer.SetWorld(XMMatrixIdentity());
	//g_PrimitiveDrawer.Begin();
	for (auto& block : barmature)
	{
		if (block->parent() != nullptr)
		{
			auto& bone = frame[block->Joints.back()->ID];
			XMVECTOR ep = bone.GblTranslation;

			auto& pbone = frame[block->parent()->Joints.back()->ID];
			XMVECTOR sp = pbone.GblTranslation;

			sp = XMVector3Transform(sp, world);
			ep = XMVector3Transform(ep, world);
			//g_PrimitiveDrawer.DrawLine(sp, ep, color);

			//XMVECTOR v = ep - sp;
			//RowVectorXf ux = block->PdGpr.uX.cast<float>();


			g_PrimitiveDrawer.DrawCylinder(sp, ep, g_DebugArmatureThinkness, color);
			g_PrimitiveDrawer.DrawSphere(ep, g_DebugArmatureThinkness * 1.5f, color);
		}
	}
	//g_PrimitiveDrawer.End();


}

void DrawGuidingVectors(const ShrinkedArmature & barmature, ArmatureFrameConstView frame, const Color & color, const Matrix4x4 & world, float thinkness = 0.015f)
{
	using DirectX::Visualizers::g_PrimitiveDrawer;
	using namespace DirectX;
	if (frame.size() == 0)
		return;
	//g_PrimitiveDrawer.SetWorld(world);
	g_PrimitiveDrawer.SetWorld(XMMatrixIdentity());
	//g_PrimitiveDrawer.Begin();
	for (auto& block : barmature)
	{
		if (block->parent() != nullptr)
		{
			auto& bone = frame[block->Joints.back()->ID];
			XMVECTOR ep = bone.GblTranslation;

			auto& pbone = frame[block->parent()->Joints.back()->ID];
			XMVECTOR sp = pbone.GblTranslation;

			sp = XMVector3Transform(sp, world);
			ep = XMVector3Transform(ep, world);
			//g_PrimitiveDrawer.DrawLine(sp, ep, color);

			g_PrimitiveDrawer.DrawCylinder(sp, ep, g_DebugArmatureThinkness, color);
			g_PrimitiveDrawer.DrawSphere(ep, g_DebugArmatureThinkness * 1.5f, color);
		}
	}
	//g_PrimitiveDrawer.End();


}

void DrawControllerHandle(const CharacterController& controller)
{
	using DirectX::Visualizers::g_PrimitiveDrawer;
	using namespace Math;

	XMVECTOR color = Colors::Pink;
	XMVECTOR vel_color = Colors::Navy;


	g_PrimitiveDrawer.SetWorld(XMMatrixIdentity());
	XMMATRIX world = controller.Character().GlobalTransformMatrix();

	auto& barmature = controller.ArmatureParts();
	auto& frame = controller.Character().GetCurrentFrame();

	for (auto& block : barmature)
	{
		if (block->ActiveActions.size() > 0)
		{
			auto& handle = controller.PvHandles()[block->Index];
			XMVECTOR ep = handle.first;

			auto& pbone = frame[block->parent()->Joints.back()->ID];
			XMVECTOR sp = pbone.GblTranslation;
			ep = sp + ep;

			sp = XMVector3Transform(sp, world);
			ep = XMVector3Transform(ep, world);

			g_PrimitiveDrawer.DrawCylinder(sp, ep, g_DebugArmatureThinkness, color);
			g_PrimitiveDrawer.DrawSphere(ep, g_DebugArmatureThinkness * 1.5f, color);
			sp = ep;
			ep = handle.second;
			ep = XMVector3TransformNormal(ep, world);
			ep = sp + ep;
			g_PrimitiveDrawer.DrawCylinder(sp, ep, g_DebugArmatureThinkness, vel_color);
			g_PrimitiveDrawer.DrawCone(ep, ep - sp, g_DebugArmatureThinkness * 5, g_DebugArmatureThinkness * 3, vel_color);
		}
	}
}

void PlayerProxy::Render(IRenderContext * context, DirectX::IEffect* pEffect)
{
	Bone charaFrame[100];

	if (g_DebugLocalMotion && g_DebugView)
	{
		for (auto& controller : m_Controllers)
		{
			if (!controller.IsReady)
				continue;
			auto& chara = controller.Character();
			auto& action = controller.Character().Behavier()[g_DebugLocalMotionAction[chara.Name]];
			action.GetFrameAt(charaFrame, current_time);
			auto world = chara.GlobalTransformMatrix();
			DrawArmature(chara.Armature(), charaFrame, DirectX::Colors::LimeGreen.v, world, g_DebugArmatureThinkness / chara.GetGlobalTransform().Scale.x);
			DrawControllerHandle(controller);
		}
	}

	if (!m_pSelector || !m_pSelector->Get()) return;
	auto& player = *m_pSelector->Get();

	Color color = DirectX::Colors::Yellow.v;

	if (player.IsAvailable())
	{
		const auto& frame = player.PeekFrame();

		if (IsMapped())
			color.A(0.3f);

		DrawArmature(player.GetArmature(), frame, reinterpret_cast<const Color*>(HumanBoneColors));
	}

	// IsMapped() && 
	if (IsMapped() && g_DebugView)
	{
		//auto& controller = this->CurrentController().Character();
		for (auto& controller : m_Controllers)
		{
			if (!controller.IsReady)
				continue;

			auto& chara = controller.Character();
			DrawControllerHandle(controller);
		}

	}

}

void XM_CALLCONV PlayerProxy::UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection)
{
	DirectX::Visualizers::g_PrimitiveDrawer.SetView(view);
	DirectX::Visualizers::g_PrimitiveDrawer.SetProjection(projection);
}

void PlayerProxy::SetPlayerSelector(const sptr<IPlayerSelector>& playerSelector) {
	m_pSelector = playerSelector;

	auto fReset = std::bind(&PlayerProxy::ResetPlayer, this, placeholders::_1, placeholders::_2);
	m_pSelector->SetPlayerChangeCallback(fReset);

	auto fFrame = std::bind(&PlayerProxy::StreamPlayerFrame, this, placeholders::_1, placeholders::_2);
	m_pSelector->SetFrameCallback(fFrame);
}

KinectVisualizer::KinectVisualizer()
{
	pKinect = Devices::KinectSensor::GetForCurrentView();
}

bool KinectVisualizer::IsVisible(const DirectX::BoundingGeometry & viewFrustum) const
{
	return true;
}

void KinectVisualizer::Render(IRenderContext * context, DirectX::IEffect* pEffect)
{
	auto &players = pKinect->GetTrackedBodies();
	using DirectX::Visualizers::g_PrimitiveDrawer;

	for (auto& player : players)
	{
		if (player.IsTracked())
		{
			const auto& frame = player.PeekFrame();

			DrawArmature(*player.BodyArmature, frame, DirectX::Colors::LimeGreen.v);
		}
	}
}

void XM_CALLCONV KinectVisualizer::UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection)
{
	DirectX::Visualizers::g_PrimitiveDrawer.SetView(view);
	DirectX::Visualizers::g_PrimitiveDrawer.SetProjection(projection);
}

RenderFlags Causality::KinectVisualizer::GetRenderFlags() const
{
	return RenderFlags::SpecialEffects;
}

