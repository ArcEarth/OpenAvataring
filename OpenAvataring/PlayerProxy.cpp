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
#include "Causality\KinectSensor.h"
#include "Causality\LeapMotion.h"
#include "Causality\MatrixVisualizer.h"
#include "Causality\AssetDictionary.h"
#include "Causality\FloatHud.h"
//					When this flag set to true, a CCA will be use to find the general linear transform between Player 'Limb' and Character 'Limb'

//float				g_NoiseInterpolation = 1.0f;
bool g_AutomaticTurnOffDependencyControl = true;
bool g_ManualRevampControl = false;
int  g_MaxTrackerCount = 2;

using namespace Causality;
using namespace Eigen;
using namespace std;
using namespace ArmaturePartFeatures;

//MatrixXf g_Testing(100000, 10000);


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

#define rgb(R,G,B) {(R##.0f / 255.0f), (G##.0f / 255.0f),(B##.0f / 255.0f), 1.0f}

static const DirectX::Color PartColorList[] = {
	rgb(26, 188, 156),
	rgb(46, 204, 113),
	rgb(52, 152, 219),
	rgb(155, 89, 182),
	rgb(52, 73, 94),
	rgb(241, 196, 15),
	rgb(230, 126, 34),
	rgb(231, 76, 60)
};
#undef rgb

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


void SetGlowBoneColor(CharacterGlowParts* glow, const ShrinkedArmature & sparts, const array_view<const Color> &colors, const CharacterController& controller, float transparency = 1.0f);

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

	auto metric = m_CyclicInfo.StreamFrame(m_pushFrame);

	if (!metric.BufferingReady || metric.ConfidenceReady)
		m_recentMetric = metric;

	if (metric.MetricReady/* && (m_mapTask.empty() || m_mapTask.is_done())*/)
	{
		SelectCharacterAsync();
	}
}

void PlayerProxy::DisplayRecentMotionMetric(const CyclicStreamClipinfo::RecentFrameResolveResult &metric)
{
	std::cout << "Start selection progress with Metric : " << endl
		<< "Frequency = " << metric.Frequency << endl
		<< "Periodic Support = " << metric.Support << endl
		<< "Kinetic Energy = " << metric.Energy << " (" << metric.AnotherEnergy << ")" << endl
		<< "Periodic Confidence = " << metric.PeriodicConfidence << endl
		<< "Static Pose Confidence = " << metric.StaticConfidence << endl
		<< "Energies = " << m_CyclicInfo.AsFacade().GetAllPartsEnergy() << endl;

}

bool PlayerProxy::SelectCharacterAsync(RecentAcrtionBehavier source, bool recaculateMetric)
{
	if (!m_mapTaskOnGoing || m_mapTask.is_done())
	{
		m_mapTaskOnGoing = true;
		m_mapTask = concurrency::create_task([this, source, recaculateMetric]() {
			try
			{
				auto metric = this->m_recentMetric;

				if (recaculateMetric)
					//if (source == CharacteSelectionSource_RecentPeriodAction)
					metric = this->m_CyclicInfo.AnaylzeRecentAction(source);
				//else
				//	this->m_CyclicInfo.AnaylzeRecentPose();

				DisplayRecentMotionMetric(metric);

				auto idx = SelectCharacter(source);
			}
			catch (const std::exception& excp)
			{
				std::cout << "[Error] Exception in Selecting Characters : " << excp.what() << endl;
			}
			m_mapTaskOnGoing = false;
		});
		return true;
	}

	std::cout << "[Error] Mapping Tasks on going" << endl;
	return false;
}

void PlayerProxy::ResetPlayer(IArmatureStreamAnimation * pOld, IArmatureStreamAnimation * pNew)
{
	StopUpdateThread();
	ResetSelection(false);
	//SetActiveController(-1);

	if (!pOld || (pNew && &pNew->GetArmature() != &pOld->GetArmature()))
	{
		ResetPlayerArmature(&pNew->GetArmature());
	}
	else if (!pNew)
	{
		ResetPlayerArmature(nullptr);
	}

	m_CyclicInfo.ResetStream();
	m_CyclicInfo.EnableCyclicMotionDetection(true);
	m_updateTime = 0;
	m_updateCounter = 0;

	m_pbValueFilter0.Reset();
	m_pbValueFilter1.Reset();
	m_pbCenterFilter.Reset();


	if (pNew)
		StartUpdateThread();
}

namespace Causality
{


	static void ColorCodeArmature(const ShrinkedArmature& parts, std::vector<Color, DirectX::XMAllocator>& boneColors)
	{
		const auto& palette = PartColorList;
		auto & armature = parts.Armature();

		boneColors.resize(armature.size());
		std::vector<int> partColorIdx(parts.size(), -1);

		int cdx = 0;
		for (auto part : parts)
		{
			int pid = part->Index;
			auto sp = part->SymetricPair;
			int spid = part->SymetricPair ? part->SymetricPair->Index : -1;

			if (partColorIdx[pid] == -1)
				partColorIdx[pid] = cdx++;

			if (sp)
				partColorIdx[spid] = partColorIdx[pid];

		}

		for (auto part : parts)
		{
			auto pcolor = palette[partColorIdx[part->Index] % std::size(palette)];
			for (auto& joint : part->Joints)
				boneColors[joint->ID] = pcolor;
		}
	}
}

void PlayerProxy::ResetPlayerArmature(const IArmature* playerArmature)
{
	m_pPlayerArmature = playerArmature;
	if (m_pPlayerArmature)
		InitializeShrinkedPlayerArmature();

	cout << "Initializing Cyclic Info..." << endl;
	m_CyclicInfo.Initialize(*m_pParts, time_seconds(g_PlayerGesturePeriodMin), time_seconds(g_PlayerGesturePeriodMax), 30, 0);
	cout << "Cyclic Info Initited!" << endl;

	{
		ResetSelection(false);
		std::lock_guard<std::mutex> guad(m_controlMutex);
		for (auto& controller : m_Controllers)
		{
			controller.SetBinding(uptr<ArmatureTransform>());
		}
	}
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
	m_EnableOverShoulderCam(false),
	m_DefaultCameraFlag(true),
	m_updateCounter(0),
	m_updateTime(0),
	m_pParts(new ShrinkedArmature()),
	m_pPlayerArmature(nullptr),
	m_selectionMode(SelectionMode_Filtering)
{
	//ResetPlayerArmature(TrackedBody::BodyArmature.get());
	Register();
	m_stopUpdate = true;
	m_IsInitialized = true;
	m_charaFrame.resize(256);

	m_pbCenterFilter.SetUpdateFrequency(&m_updateFreqency);
	m_pbValueFilter0.SetUpdateFrequency(&m_updateFreqency);
	m_pbValueFilter1.SetUpdateFrequency(&m_updateFreqency);
	m_cameraStablizer.SetUpdateFrequency(&m_updateFreqency);

	m_pbCenterFilter.SetCutoffFrequency(1.0f);
	m_pbValueFilter0.SetCutoffFrequency(1.0f);
	m_pbValueFilter1.SetCutoffFrequency(1.0f);
	m_cameraStablizer.SetCutoffFrequency(3.0f);

}

void PlayerProxy::InitializeShrinkedPlayerArmature()
{
	if (g_EnableDebugLogging >= 1)
	{
		cout << "Player Armature: " << endl;
		int i = 0;
		auto& armature = *m_pPlayerArmature;
		for (auto& j : armature.joints())
		{
			cout << "Joint [" << j.ID << "] = " << j.Name;
			if (j.MirrorJoint != nullptr)
			{
				cout << " <--> " << j.MirrorJoint->Name << endl;
			}
			else
				cout << endl;
		}
	}

	m_pParts->SetArmature(*m_pPlayerArmature);

	if (g_EnableDebugLogging >= 1)
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

		cout << "Armature Proportions : " << endl;
		for (auto pPart : *m_pParts)
		{
			auto& part = *pPart;
			cout << "Part " << part.Joints << " = ";
			cout << part.ChainLength << '|' << part.LengthToRoot << endl;
		}
	}

	ColorCodeArmature(*m_pParts, m_boneColors);

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
	m_recentMetric = CyclicStreamClipinfo::RecentFrameResolveResult();
	m_recentMetric.BufferingReady = true;
	m_recentMetric.BufferingProgress = 0.0f;
	m_recentMetric.StaticConfidence = 0.0f;
	m_recentMetric.PeriodicConfidence = 0.0f;
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

		auto mat = pChara->FirstChildOfType<MatrixVisualizer>();
		if (mat == nullptr)
		{
			mat = new MatrixVisualizer();
			mat->Scene = this->Scene;
			IsometricTransform lt;
			lt.Translation = Vector3(0, 0.5, 0);
			mat->SetLocalTransform(lt);
			pChara->AddChild(mat);
		}

		auto canvas = pChara->FirstChildOfType<SpriteCanvas>();
		if (canvas == nullptr)
		{
			canvas = new SpriteCanvas();
			canvas->Scene = this->Scene;
			canvas->SetBackground(DirectX::Colors::Red.v);
			canvas->CreateDeviceResources(this->Scene->GetRenderDevice(), 200, 60);
			std::shared_ptr<TextBlock> nameTag(new TextBlock());
			nameTag->SetText(pChara->Name);
			nameTag->CreateDeviceResources(canvas->Get2DFactory(), canvas->GetTextFactory());
			canvas->AddChild(std::static_pointer_cast<HUDElement>(nameTag));
			canvas->SetVisiability(true);
			IsometricTransform lt;
			lt.Translation = Vector3(0, 0.5, 0);
			canvas->SetLocalTransform(lt);
			pChara->AddChild(canvas);
			canvas->SetScale(Vector3(1.0f));
		}
	}
}

void PlayerProxy::Parse(const ParamArchive * store)
{
	auto sel = GetFirstChildArchive(store, "player_controller.selector");
	sel = GetFirstChildArchive(sel);
	string name = GetArchiveName(sel);

	//g_Testing.setRandom();

	PlayerSelectorBase::SelectionMode mode;
	unsigned umode = PlayerSelectorBase::ClosestStickly;
	GetParam(sel, "mode", umode);
	mode = (PlayerSelectorBase::SelectionMode)umode;
	sptr<IPlayerSelector> pSelector;

	string trail_particle;
	GetParam(store, "trail_particle", trail_particle);
	if (!trail_particle.empty())
	{
		auto& assets = Scene->Assets();
		if (trail_particle[0] == '{')
			m_trailVisual = dynamic_cast<Texture2D*>(assets.GetTexture(trail_particle.substr(1, trail_particle.size() - 2)));
		else
			m_trailVisual = dynamic_cast<Texture2D*>(assets.LoadTexture("character_trail_visual", trail_particle));
	}


	if (name == "kinect_source")
	{
		auto pKinect = Devices::KinectSensor::GetForCurrentView();
		pSelector = make_shared<KinectPlayerSelector>(pKinect.get(), mode);
		pKinect->Start();
	}
	else if (name == "leap_source")
	{
		auto pLeap = Devices::LeapSensor::GetForCurrentView();
		pSelector = make_shared<LeapPlayerSelector>(pLeap.get(), mode);
		pLeap->Start();
	}
	else if (name == "virutal_character_source")
	{
	}

	if (pSelector)
		SetPlayerSelector(pSelector);
}

void SetGlowBoneColorPartPair(CharacterGlowParts * glow, int Jx, int Jy, const array_view<const Color> &colors, float transparency, const Causality::ShrinkedArmature & sparts, const Causality::ShrinkedArmature & cparts);

void SetGlowBoneColor(CharacterGlowParts* glow, const Causality::ShrinkedArmature & sparts, const array_view<const Color> &colors, const CharacterController& controller, float transparency)
{
	auto pTrans = &controller.Binding();
	auto pCcaTrans = dynamic_cast<const BlockizedCcaArmatureTransform*>(pTrans);
	auto pPartTrans = dynamic_cast<const PartilizedTransformer*>(pTrans);

	auto& cparts = controller.ArmatureParts();

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
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, transparency, sparts, cparts);
		}
	}
	else if (pPartTrans)
	{
		for (auto& tp : pPartTrans->ActiveParts)
		{
			auto Jx = tp.SrcIdx, Jy = tp.DstIdx;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, transparency, sparts, cparts);
		}
		for (auto& tp : pPartTrans->DrivenParts)
		{
			auto Jx = tp.SrcIdx, Jy = tp.DstIdx;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, transparency, sparts, cparts);
		}
		for (auto& tp : pPartTrans->AccesseryParts)
		{
			auto Jx = tp.SrcIdx, Jy = tp.DstIdx;
			SetGlowBoneColorPartPair(glow, Jx, Jy, colors, transparency, sparts, cparts);
		}
	}

}

void SetGlowBoneColorPartPair(Causality::CharacterGlowParts * glow, int Jx, int Jy, const array_view<const Color> &colors, float transparency, const Causality::ShrinkedArmature & sparts, const Causality::ShrinkedArmature & cparts)
{
	using namespace Math;
	XMVECTOR color;
	if (Jx == NoInputParts)
		color = Colors::Transparent;
	else if (Jx == ActiveAndDrivenParts)
		color = Colors::DarkGray;
	else if (Jx == ActiveParts)
		color = Colors::Gray;
	else if (Jx >= 0)
		color = colors[sparts[Jx]->Joints.front()->ID];

	using namespace DirectX;
	color = XMVectorSetW(color, 0.5f * transparency);
	for (auto joint : cparts[Jy]->Joints)
	{
		glow->SetBoneColor(joint->ID, color);
	}
}

void PlayerProxy::ResetSelection(bool enableGlow)
{
	{
		std::lock_guard<std::mutex> guard(m_controlMutex);
		for (auto& ctrl : m_Controllers)
		{
			ctrl.IsSelected = true;
			ctrl.SetBinding(nullptr);
			ctrl.CharacterScore = enableGlow ? 1.0f : .0f;
			ResetChracterGlow(ctrl);
			DeactivateController(ctrl);
		}
	}
	SetActiveController(-1);
}

template <typename Func>
auto at_scope_exit(Func func)
{
	struct scope_guard {
		Func func;
		~scope_guard()
		{
			func();
		}
	};

	return scope_guard{ func };
};

void PlayerProxy::SetActiveController(int idx)
{
	//if (!std::lock(m_controlMutex))
	//	return;

	if (idx >= 0)
		idx = idx % m_Controllers.size();

	if (idx == -1)
		ResetPrimaryCameraPoseToDefault();

	if (idx == m_CurrentIdx)
		return;

	StopUpdateThread();
	std::lock_guard<std::mutex> guard(m_controlMutex);

	auto sguard = at_scope_exit([this]() {
		this->StartUpdateThread();
	});

	for (auto& c : m_Controllers)
	{
		if (c.ID != idx)
		{
			DeactivateController(c);
		}
	}

	if (m_CurrentIdx != idx)
	{
		m_CurrentIdx = idx;
		if (m_CurrentIdx != -1)
		{
			return ActivateController(GetController(m_CurrentIdx));
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
			SetGlowBoneColor(glow, *m_pParts, m_boneColors, controller);
			glow->SetEnabled(!g_DebugView);
		}
	}


	//else
	//	StopUpdateThread();

}

void PlayerProxy::ActivateController(CharacterController & controller)
{
	//auto& controller = GetController(m_CurrentIdx);
	auto& chara = controller.Character();

	if (!g_DebugLocalMotion && !chara.CurrentActionName().empty())
	{
		g_DebugLocalMotionAction[chara.Name] = chara.CurrentActionName();
		chara.StopAction();
	}

	if (!(m_pSelector && m_pSelector->Get()))
		return;

	auto &player = *m_pSelector->Get();
	auto& frame = player.PeekFrame();
	auto pose = frame[m_pPlayerArmature->root()->ID];
	controller.SetReferenceSourcePose(pose);

	auto selected = controller.IsSelected;
	controller.IsSelected = true;
	auto temp = controller.CharacterScore;
	controller.CharacterScore = 1.0f;

	ResetChracterGlow(controller);
	controller.IsSelected = selected;
	controller.CharacterScore = temp;

	chara.ShowBoundingGeometry(true);
	chara.EnabeAutoDisplacement(g_UsePersudoPhysicsWalk);
}

void PlayerProxy::DeactivateController(Causality::CharacterController & c)
{
	int idx = c.ID;
	auto& chara = c.Character();

	if (!g_DebugLocalMotion && !g_DebugLocalMotionAction[chara.Name].empty())
	{
		auto& action = g_DebugLocalMotionAction[chara.Name];
		chara.StartAction(action);
		g_DebugLocalMotionAction[chara.Name] = "";
	}

	if (c.IsSelected && c.IsBindingReady())
	{
		chara.SetPosition(c.CMapRefPos);
		chara.SetOrientation(c.CMapRefRot);
		chara.EnabeAutoDisplacement(false);
	}

	ResetChracterGlow(c);
	chara.ShowBoundingGeometry(false);

	auto matvis = chara.FirstChildOfType<MatrixVisualizer>();
	if (matvis)
		matvis->SetEnabled(false);
}

namespace std
{
	template <typename T>
	static inline T clamp(T value, T min_val, T max_val)
	{
		return std::min(std::max(value, min_val), max_val);
	}
}

const ArmatureFrameAnimation* GetCharacterSelectiveAction(const CharacterObject& chara)
{
	if (chara.CurrentAction() != nullptr)
		return chara.CurrentAction();
	else
	{
		auto name = g_DebugLocalMotionAction[chara.Name];
		auto& behavier = chara.Behavier();
		if (behavier.Contains(name))
		{
			return &behavier[name];
		}
		else
		{
			name = DefaultAnimationSet;
			if (behavier.Contains(name))
			{
				return &behavier[name];
			}
			else
			{
				return nullptr;
			}
		}
	}
}

int PlayerProxy::SelectCharacter(RecentAcrtionBehavier source)
{

	if (!m_pSelector || !m_pSelector->Get() || 
			std::all_of(BEGIN_TO_END(m_Controllers), [](const auto& ctr)->bool
			{ return !ctr.IsReady; }))
		return -1;

	auto& player = *m_pSelector->Get();
	CharacterController* pControl = nullptr;
	{
		if (g_EnableDebugLogging > 1)
			std::cout << "Aquiring Facade Lock" << endl;
		std::lock_guard<std::mutex> guard(m_CyclicInfo.AqucireFacadeMutex());
		if (g_EnableDebugLogging > 1)
			std::cout << "FacadeLock Aquired" << endl;

		//std::this_thread::sleep_for(std::chrono::seconds(1));
		std::vector<float> scores(m_Controllers.size());

		std::vector<std::reference_wrapper<CharacterController>>
			activeControllers;

		for (auto& controller : m_Controllers)			//? <= 5 character
														//concurrency::parallel_for_each(BEGIN_TO_END(m_Controllers),[this](CharacterController& controller)
		{
			auto& chara = controller.Character();
			auto action = GetCharacterSelectiveAction(chara);

			if (!controller.IsReady || !controller.IsSelected 
				|| !action
				|| !action->IsCyclic)
			{
				controller.CharacterScore = -10000.0f;
				continue;
			}

			activeControllers.push_back(controller);
		}

		if (g_EnableDebugLogging >= 1)
		{
			for (auto ctrref : activeControllers)			//? <= 5 character
			//concurrency::parallel_for_each(BEGIN_TO_END(m_Controllers),[this](CharacterController& controller)
			{
				auto& controller = ctrref.get();
				auto action = GetCharacterSelectiveAction(controller.Character());

				controller.CreateControlBinding(m_CyclicInfo.AsFacade(), action->Name);
			}
		}
		else
		{
			concurrency::parallel_for_each(BEGIN_TO_END(m_Controllers), [this](std::reference_wrapper<CharacterController> ctrref)
			{
				auto& controller = ctrref.get();
				auto action = GetCharacterSelectiveAction(controller.Character());

				controller.CreateControlBinding(m_CyclicInfo.AsFacade(), action->Name);
			});
		}
		//);

		if (g_EnableDebugLogging > 1)
			std::cout << "FacadeLock Releasing" << endl;
	}


	auto ctritr = std::max_element(BEGIN_TO_END(m_Controllers), [](const auto& c0, const auto& c1)
	{
		return c0.CharacterScore < c1.CharacterScore;
	});

	auto maxScore = ctritr->CharacterScore;
	if (ctritr->IsReady && ctritr->CharacterScore > .0f)
	{
		int selectionCount = 0;
		for (auto& ctr : m_Controllers)
		{
			auto& chara = ctr.Character();
			auto score = ctr.CharacterScore / maxScore;
			score = std::clamp(score, 0.0f, 1.0f);
			ctr.CharacterScore = score;
			ctr.IsSelected = score > g_CharacterSelectionLikilihoodThreshold;

			selectionCount += ctr.IsSelected;

			ResetChracterGlow(ctr);

			if (m_selectionMode == SelectionMode_Filtering)
			{
				if (ctr.IsSelected)
					ActivateController(ctr);
				else
					DeactivateController(ctr);
			}
		}

		if (selectionCount == 0 && !IsMapped())
			ResetSelection(false);

		if (selectionCount == 1 || m_selectionMode == SelectionMode_MostLikilily)
		{
			pControl = &(*ctritr);

			// Disable re-matching when the controller has not request
			//m_CyclicInfo.EnableCyclicMotionDetection(false);
		}
		else
		{
			m_CyclicInfo.ResetStream();
		}
	}

	if (!pControl) return -1;

	SetActiveController(pControl->ID);

	return pControl->ID;
}

void PlayerProxy::ResetChracterGlow(CharacterController & ctr)
{
	auto& chara = ctr.Character();
	auto glow = chara.FirstChildOfType<CharacterGlowParts>();

	auto score = ctr.CharacterScore;
	chara.SetOpticity(score);

	if (score > g_CharacterSelectionLikilihoodThreshold)
	{
		if (glow)
		{
			glow->SetEnabled(true);
			SetGlowBoneColor(glow, *m_pParts, m_boneColors, ctr, score);
		}
	}
	else
	{
		if (glow)
			glow->SetEnabled(false);
	}
}




bool PlayerProxy::IsMapped() const {

	//return m_CurrentIdx >= 0;
	return (m_selectionMode == SelectionMode_MostLikilily && m_CurrentIdx >= 0) ||
		(m_selectionMode == SelectionMode_Filtering && 
			std::any_of(m_Controllers.begin(), m_Controllers.end(),
				[](const CharacterController& c) {
					return c.IsSelected && c.IsBindingReady();
				})
		);
}

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

std::vector<CharacterController*> PlayerProxy::GetAllSelectedControllers()
{
	std::vector<CharacterController*> ctrls;
	if (m_selectionMode == SelectionMode_MostLikilily && m_CurrentIdx >= 0)
		ctrls.push_back(&CurrentController());
	else
		for (auto& c : m_Controllers)
		{
			if (c.IsSelected && c.IsBindingReady())
				ctrls.push_back(&c);
		}
	return ctrls;
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
	else if (e.Key == 'O')
	{
		g_EnableDependentControl = !g_EnableDependentControl;
		cout << "[Key] Enable Dependency Control = " << g_EnableDependentControl << endl;
	}
	else if (e.Key == 'C')
	{
		m_EnableOverShoulderCam = !m_EnableOverShoulderCam;
		//g_UsePersudoPhysicsWalk = m_EnableOverShoulderCam;
		cout << "[Key] Over Shoulder Camera Mode = " << m_EnableOverShoulderCam << endl;
		//cout << "Persudo-Physics Walk = " << g_UsePersudoPhysicsWalk << endl;
	}
	else if (e.Key == 'V')
	{
		g_UsePersudoPhysicsWalk = !g_UsePersudoPhysicsWalk;
		cout << "[Key] Persudo-Physics Walk = " << g_UsePersudoPhysicsWalk << endl;
		for (auto& controller : m_Controllers)
		{
			controller.Character().EnabeAutoDisplacement(g_UsePersudoPhysicsWalk && controller.ID == m_CurrentIdx);
		}
	}
	else if (e.Key == 'Z')
	{
		static const char* modeName[] = {
			"Selection Mode : NONE",
			"Multiple Selection",
			"Unique Selection",
		};
		m_selectionMode = m_selectionMode == SelectionMode_Filtering ? SelectionMode_MostLikilily : SelectionMode_Filtering;
		cout << "[Key] Selection Mode Changed ==> [" << modeName[m_selectionMode] << ']' << endl;
	}
	else if (e.Key == 'E')
	{
		cout << "[Key] Implicit Action Selection ==> ENABLED" << endl;
		m_CyclicInfo.EnableCyclicMotionDetection();
		g_ManualRevampControl = true;
	}
	else if (e.Key == 'R')
	{
		cout << "[Key] Implicit Action Selection ==> DISABLED" << endl;
		m_CyclicInfo.EnableCyclicMotionDetection(false);
		g_ManualRevampControl = true;
	}
	//else if (e.Key == 'M')
	//{
	//	g_MirrowInputX = !g_MirrowInputX;
	//	cout << "Kinect Input Mirrowing = " << g_MirrowInputX << endl;
	//	m_pKinect->EnableMirrowing(g_MirrowInputX);
	//}
	//else if (e.Key == VK_BACK)
	//{
	//	m_CyclicInfo.ResetStream();
	//	m_DefaultCameraFlag = true;
	//}
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
	else if (e.Key == 'P')
	{
		cout << "[Key] Reset Camera Pose" << endl;
		ResetPrimaryCameraPoseToDefault();
	}
	else if (e.Key == VK_RETURN || e.Key == VK_TAB)
	{
		static const char* sourName[] =
		{
			"Auto",
			"FreezedPose",
			"PeriodMotion",
		};
		auto source = e.Modifier & KeyModifiers::Mod_Control || e.Key == VK_TAB ?
			RecentActionBehavier_FreezedPose :
			RecentActionBehavier_PeriodMotion;

		cout << "[Key] Manual Triggerd Selection : " << sourName[source] << endl;

		if (!SelectCharacterAsync(source, true))
		{
			cout << "Selection is already going on..." << endl;
		}
	}
	else if (e.Key == VK_BACK)
	{
		cout << "[Key] Reset Selection & Control Mapping" << endl;
		m_DefaultCameraFlag = true;
		ResetSelection(false);
		m_CyclicInfo.ResetStream();
		m_CyclicInfo.EnableCyclicMotionDetection();
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

		//g_RevampLikilyhoodThreshold = 0.5;
		//g_RevampLikilyhoodTimeThreshold = 1.0;

		if (g_ForceRemappingAlwaysOn)
			m_CyclicInfo.EnableCyclicMotionDetection();

		if (IsMapped() && m_controlMutex.try_lock())
		{
			std::lock_guard<std::mutex> guard(m_controlMutex, std::adopt_lock);
			//cout << "getting mutext" << endl;

			auto controllers = GetAllSelectedControllers();

			float lik = 1.0f;

			int sz = controllers.size();

			if (g_AutomaticTurnOffDependencyControl)
			{
				g_EnableDependentControl = sz <= g_MaxTrackerCount;
			}

			if (sz > 1)
			{

				#pragma omp parallel for
				for (int i = 0; i < sz; ++i)
				{
					auto& controller = *controllers[i];
					float lik = controller.UpdateTargetCharacter(frame, lastFrame, dt);
				}
			} else
			{ 
				for (int i = 0; i < sz; ++i)
				{
					auto& controller = *controllers[i];
					float lik = controller.UpdateTargetCharacter(frame, lastFrame, dt);
				}
			}

			// Check if we need to "Revamp" Control Binding
			if (!g_ManualRevampControl)
			if (sz == 1)
			{
				if (lik < g_RevampLikilyhoodThreshold)
				{
					m_LowLikilyTime += dt;
					m_CyclicInfo.EnableCyclicMotionDetection(m_LowLikilyTime > g_RevampLikilyhoodTimeThreshold);
				}
				else
				{
					m_CyclicInfo.EnableCyclicMotionDetection(false);
					m_LowLikilyTime = 0;
				}
			} else if (sz > 1)
			{
				m_CyclicInfo.EnableCyclicMotionDetection(g_AllowsRevampInMultipleSelection);
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

	m_updateFreqency = (1.0f / time_delta.count());

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
		auto& action = chara.Behavier()[actionName];
		auto& target_frame = chara.MapCurrentFrameForUpdate();
		ArmatureFrame frame(chara.Armature().bind_frame());

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
		//target_frame = m_charaFrame;
		//AddNoise(frame, .1f);

		target_frame = frame;
		controller.SelfBinding().Transform(target_frame, frame, last_frame, time_delta.count());
	}
}

void PlayerProxy::UpdatePrimaryCameraForTrack()
{
	auto& camera = *this->Scene->PrimaryCamera();
	auto& cameraPos = dynamic_cast<SceneObject&>(camera);

	if (m_CurrentIdx == -1)
		return;

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
		m_cameraStablizer.Reset();
	}

	XMVECTOR pos = m_cameraStablizer.Apply(chara.GetPosition());

	cameraPos.SetPosition(pos + XMVector3Rotate(XMVectorMultiplyAdd(ext, XMVectorSet(-2.0f, 2.0f, -2.0f, 0.0f), XMVectorSet(-0.5f, 0.5, -0.5, 0)), chara.GetOrientation()));
	camera.GetView()->FocusAt(pos + XMVector3Rotate(XMVectorMultiplyAdd(ext, XMVectorSet(-2.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(-0.5f, 0.5, -0.5, 0)), chara.GetOrientation()), g_XMIdentityR1.v);
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

void XM_CALLCONV DrawParticle(DirectX::SpriteBatch & sprites, DirectX::XMVECTOR particle, const DirectX::XMVECTOR &color, DirectX::CXMMATRIX worldView, DirectX::CXMMATRIX proj, const D3D11_VIEWPORT & vp, ID3D11ShaderResourceView * pTrajectoryVisual)
{
	using namespace DirectX;
	particle = XMParticleProjection(particle, worldView, proj, vp);
	float depth = _DXMEXT XMVectorGetZ(particle);
	particle = _DXMEXT XMVectorSwizzle<0, 1, 3, 3>(particle);
	Vector4 fp = particle;
	RECT rect;
	rect.left = fp.x - fp.z;
	rect.right = fp.x + fp.z;
	rect.top = fp.y - fp.w;
	rect.bottom = fp.y + fp.w;
	sprites.Draw(pTrajectoryVisual, rect, nullptr, color, 0, DirectX::XMFLOAT2(0, 0), DirectX::SpriteEffects::SpriteEffects_None, depth);
}

void XM_CALLCONV DrawControllerHandle(CharacterController& controller, DirectX::Color* colors, FXMMATRIX view, CXMMATRIX proj, const D3D11_VIEWPORT& vp, ID3D11ShaderResourceView* pTrajectoryVisual = nullptr)
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
			if (colors)
				color = XMLoad(colors[block->Index]);
			auto handle = controller.GetPvHandle(block->Index);
			auto trajectory = controller.GetPvHandleTrajectory(block->Index);

			XMVECTOR ep = handle.first;

			auto& pbone = frame[block->parent()->Joints.back()->ID];
			XMVECTOR sp = pbone.GblTranslation;
			ep = sp + ep;

			// forward trajectory's ending to it's local coordinate
			if (!trajectory.empty())
				*trajectory.begin() = ep;

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


	if (!pTrajectoryVisual) return;
	auto& sprites = *g_PrimitiveDrawer.GetSpriteBatch();
	sprites.Begin(DirectX::SpriteSortMode_Deferred, g_PrimitiveDrawer.GetStates()->NonPremultiplied());
	world = XMMatrixMultiply(world, view);
	float particleRadius = g_DebugArmatureThinkness * 5;

	float transparencyDecay = 1.0f / 30.0f;
	for (auto& block : barmature)
	{
		float transparency = 1.0f;
		if (block->ActiveActions.size() > 0)
		{
			if (colors)
				color = XMLoad(colors[block->Index]);
			color = XMVectorSaturate(color + XMVectorReplicate(0.3f));
			auto trajectory = controller.GetPvHandleTrajectory(block->Index);
			for (auto& p : trajectory)
			{

				XMVECTOR particle = XMLoad(p);
				particle = XMVectorSetW(particle, particleRadius * (2.0f - 1.7 * transparency * transparency));
				color = XMVectorSetW(color, transparency);
				DrawParticle(sprites, particle, color, world, proj, vp, pTrajectoryVisual);
				transparency -= transparencyDecay;
				transparency = fmax(transparency, .0f);
			}
		}
	}

	sprites.End();
}

void DrawProgressBar(Vector3 center, float width, float radius, float progress, Color foreground, Color background)
{
	Vector3 hext = Vector3(0.5f * width, 0, 0);
	Vector3 left = center - hext;
	Vector3 right = center + hext;
	Vector3 val = Vector3::Lerp(left, right, progress);
	auto& drawer = DirectX::Visualizers::g_PrimitiveDrawer;

	if (progress > 0.01f)
		drawer.DrawCylinder(left, val, radius, foreground);
	if (progress < 0.99f)
		drawer.DrawCylinder(val + hext * 0.01f, right, radius, background);
}

void PlayerProxy::Render(IRenderContext * context, DirectX::IEffect* pEffect)
{
	//Bone charaFrame[100];

	auto view = DirectX::Visualizers::g_PrimitiveDrawer.GetView();
	auto proj = DirectX::Visualizers::g_PrimitiveDrawer.GetProjection();
	D3D11_VIEWPORT viewport;
	uint32_t numViewport = 1;
	context->RSGetViewports(&numViewport, &viewport);

	if (g_DebugLocalMotion && g_DebugView)
	{
		for (auto& controller : m_Controllers)
		{
			if (!controller.IsReady)
				continue;
			auto& chara = controller.Character();
			auto& action = controller.Character().Behavier()[g_DebugLocalMotionAction[chara.Name]];
			if (chara.Armature().size() > m_charaFrame.size())
				m_charaFrame.resize(chara.Armature().size());
			action.GetFrameAt(m_charaFrame, current_time);
			auto world = chara.GlobalTransformMatrix();
			DrawArmature(chara.Armature(), m_charaFrame, DirectX::Colors::LimeGreen.v, world, g_DebugArmatureThinkness / chara.GetGlobalTransform().Scale.x);
			DrawControllerHandle(controller, nullptr, view, proj, viewport, *m_trailVisual);
		}
	}

	if (!m_pSelector || !m_pSelector->Get()) return;
	auto& player = *m_pSelector->Get();

	Color color = DirectX::Colors::Yellow.v;

	if (player.IsAvailable() /*&& (!IsMapped() || g_ForceRemappingAlwaysOn)*/)
	{
		const auto& frame = player.PeekFrame();

		if (IsMapped())
			color.A(0.3f);

		if (frame.size() != 0)
		{
			XMMATRIX world = DirectX::XMMatrixScaling(1.0, 1.0, 1.0);
			float thinkness = g_DebugArmatureThinkness;
			DrawArmature(player.GetArmature(), frame, m_boneColors.data(), world, thinkness * 3);

			auto& drawer = DirectX::Visualizers::g_PrimitiveDrawer;
			auto highest = std::max_element(frame.begin(), frame.end(), [](const Bone& b0, const Bone& b1)
			{
				return b0.GblTranslation.y < b1.GblTranslation.y;
			}
			);

			auto metric = m_recentMetric;

			bool buffering = !metric.BufferingReady;

			Color pbbc = DirectX::Colors::Gray;

			Color pbfc = buffering ?
				DirectX::Colors::Yellow :
				DirectX::Colors::YellowGreen;


			float maxalpha = 0.8f;
			float progress = buffering ? metric.BufferingProgress : metric.PeriodicConfidence;
			progress = std::clamp(progress, .0f, 1.0f);
			float radius = 0.15f;
			float width = 1.0f;
			float alpha = maxalpha;
			pbfc.A(alpha); pbbc.A(alpha);

			Vector3 center = frame[0].GblTranslation;
			center.y = highest->GblTranslation.y + 0.3f;

			center = m_pbCenterFilter.Apply(center);
			if (!buffering)
			{
				progress = m_pbValueFilter0.Apply(progress);
				float alpha = 0.2f + maxalpha * sqrt(progress);
				pbfc.A(alpha); pbbc.A(alpha);
			}

			DrawProgressBar(center, width, radius, progress, pbfc, pbbc);

			if (!buffering)
			{
				progress = std::clamp(metric.StaticConfidence, .0f, 1.0f);
				progress = m_pbValueFilter1.Apply(progress);

				pbfc = DirectX::Colors::Cyan.v;
				float alpha = 0.2f + maxalpha * sqrt(progress);
				pbfc.A(alpha); pbbc.A(alpha);
				center += Vector3(0, 0.3f, 0);
				DrawProgressBar(center, width, radius, progress, pbfc, pbbc);
			}
		}

	}

	if (IsMapped() /*&& g_DebugView */ /*&& m_CurrentIdx != -1*/)
	{
		//auto& controller = this->CurrentController().Character();

		auto selectes = GetAllSelectedControllers();

		//auto & controller = CurrentController();
		for (auto pController : selectes)
		{
			auto& controller = *pController;

			auto& chara = controller.Character();
			auto glow = chara.FirstChildOfType<CharacterGlowParts>();

			std::vector<Color> colors(controller.ArmatureParts().size());

			for (auto& part : controller.ArmatureParts())
			{
				colors[part->Index] = glow->GetBoneColor(part->Joints.front()->ID);
			}

			auto& frame = chara.GetCurrentFrame();
			auto tran = chara.GetGlobalTransform();
			auto& drawer = DirectX::Visualizers::g_PrimitiveDrawer;
			auto highest = std::max_element(frame.begin(), frame.end(), [](const Bone& b0, const Bone& b1)
			{ return b0.GblTranslation.y < b1.GblTranslation.y; });

			Vector3 center = frame[0].GblTranslation;
			float scl = 0.2f / tran.Scale.y;

			center.y = highest->GblTranslation.y + scl * 2.0;
			Color color = DirectX::Colors::ForestGreen.v;
			color.A(0.8f);

			drawer.SetWorld(tran.TransformMatrix());
			drawer.DrawCone(center, DirectX::g_XMIdentityR1.v, -scl * 1.5, scl * 0.5, color);

			DrawControllerHandle(controller, colors.data(), view, proj, viewport, *m_trailVisual);

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

