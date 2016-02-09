#include "pch.h"
#include "PlayerSelector.h"
#include <Causality\KinectSensor.h>
#include <Causality\LeapMotion.h>
using namespace Causality;
using namespace Causality::Devices;

class PlayerSelectorBase::MergedTrackedArmature : public TrackedArmature
{
public:
	MergedTrackedArmature(const std::vector<TrackedArmature*>& players);
	~MergedTrackedArmature()
	{
		Disconnet();
	}

	void Disconnet()
	{
		int pid = 0;
		for (auto&& player : m_subarms)
		{
			m_cons[pid].con_frame.disconnect();
			m_cons[pid].con_lost.disconnect();
			player->Release();
			player = nullptr;
			++pid;
		}
		m_subarms.clear();
	}

	void ComposeFrame(_Out_ ArmatureFrameView frame);

	const IArmature& GetArmature() const {
		return m_armature;
	}

public:
	void _CollectFrame(const TrackedArmature&, const FrameType&);
	void _CollectLost(const TrackedObject&);

	void FillSubarmature(_Out_ ArmatureFrameView frame, int pid, const ArmatureFrameConstView& subframe);
protected:
	void SumriseRoot(ArmatureFrameView frame);
private:
	ArmatureFrame					 m_frameCache;
	std::vector<int>				 m_signals;
	std::vector<TrackedArmature*>	 m_subarms;
	std::vector<std::pair<int, int>> m_subarmidx;
	struct SubArmCon 
	{ 
		EventConnection con_frame, con_lost;
	};
	std::vector < SubArmCon >		 m_cons;
};

void PlayerSelectorBase::ChangeSelectionMode(SelectionMode mdoe)
{
	this->mode = mode;
	if (m_current)
		Reselect();
}

PlayerSelectorBase::PlayerSelectorBase()
	: m_current(nullptr), m_sensor(nullptr), mode(None) 
{
}

PlayerSelectorBase::~PlayerSelectorBase()
{
	fpTrackedBodyChanged = nullptr;
	fpFrameArrived = nullptr;

	ChangePlayer(nullptr);

	if (m_sensor)
	{
		con_tracked.disconnect();
		con_lost.disconnect();
	}
}

float PlayerSelectorBase::Distance(const TrackedArmature & body) const {
	return 1.0f;
}

void PlayerSelectorBase::Reset()
{
	if (fpTrackedBodyChanged)
		fpTrackedBodyChanged(m_current, nullptr);
	if (m_current)
		m_current->Release();
	m_current = nullptr;
}

void PlayerSelectorBase::ChangePlayer(TrackedArmature * pNewPlayer)
{
	auto pOld = m_current;
	con_frame.disconnect();
	m_current = pNewPlayer;

	if (m_current)
	{
		m_current->AddRef();
		if (fpFrameArrived)
			con_frame = m_current->OnFrameArrived.connect(fpFrameArrived);
	}

	if (fpTrackedBodyChanged)
		fpTrackedBodyChanged(pOld, m_current);

	if (pOld)
	{
		pOld->Release();

		if (m_oldmerged && m_oldmerged->RefCount() <= 0)
			m_oldmerged.reset();
	}
}

void PlayerSelectorBase::SetFrameCallback(const FrameEventFunctionType & callback)
{
	fpFrameArrived = callback;

	if (con_frame.connected())
		con_frame.disconnect();

	if (fpFrameArrived && m_current)
		con_frame = m_current->OnFrameArrived.connect(fpFrameArrived);
}

void PlayerSelectorBase::SetPlayerChangeCallback(const PlayerEventFunctionType & callback)
{
	fpTrackedBodyChanged = callback;
	if (m_current && fpTrackedBodyChanged)
	{
		fpTrackedBodyChanged(nullptr, m_current);
	}
}

IArmatureStreamAnimation * PlayerSelectorBase::Get() { return m_current; }

void PlayerSelectorBase::OnPlayerTracked(TrackedArmature & body)
{
	if (!m_current)
	{
		ChangePlayer(&body);
	} else if (!(mode & Sticky) && (mode == SelectionMode::Closest && Distance(body) < Distance(*m_current)))
	{
		ChangePlayer(&body);
	}
	else if (mode == MergeAll)
	{
		Reselect();
	}
}

void PlayerSelectorBase::OnPlayerLost(TrackedArmature & body)
{
	if (m_current && (body == *m_current || mode == MergeAll))
	{
		Reselect();
	}
}

void PlayerSelectorBase::Reselect()
{
	TrackedArmature *pBestPlayer = nullptr;

	GetTrackedArmatures(m_candidates);

	if (mode & Closest)
	{
		float distance = 100000;
		for (auto& player : m_candidates)
		{
			if (player->IsTracked() && (m_current == nullptr || *player != *m_current && distance > Distance(*player)))
			{
				pBestPlayer = player;
				distance = Distance(*player);
			}
		}
	}
	else if (mode == MergeAll)
	{
		if (m_candidates.size() > 0)
		{
			// we must keep the m_merged alive before PlayerChanged is called
			m_oldmerged = move(m_merged);

			if (m_candidates.size() > 1)
			{
				m_merged.reset(new MergedTrackedArmature(m_candidates));
				pBestPlayer = m_merged.get();
				//m_merged->Lost.connect([this, pBestPlayer](const TrackedObject& obj)
				//{
				//});
			}
			else
				pBestPlayer = m_candidates.front();
		}
	}
	else // Eariest tracked player
	{
		for (auto& player : m_candidates)
		{
			if (player->IsTracked() && (m_current == nullptr || *player != *m_current))
			{
				pBestPlayer = player;
				break;
			}
		}
	}

	ChangePlayer(pBestPlayer);
}


KinectPlayerSelector::KinectPlayerSelector(KinectSensor * pKinect, SelectionMode mode)
{
	Initialize(pKinect, mode);
}

KinectPlayerSelector::~KinectPlayerSelector()
{
}

float KinectPlayerSelector::Distance(const TrackedArmature & body) const
{
	auto pBody = dynamic_cast<const TrackedBody*>(&body);
	if (pBody)
		return pBody->DistanceToSensor();
	return 1.0f;
}

void KinectPlayerSelector::Initialize(Devices::KinectSensor * pKinect, SelectionMode mode)
{
	this->mode = mode;
	if (pKinect)
	{
		this->m_sensor = pKinect->GetRef();
		con_tracked =
			pKinect->OnPlayerTracked += MakeEventHandler(&KinectPlayerSelector::OnPlayerTracked, this);
		con_lost =
			pKinect->OnPlayerLost += MakeEventHandler(&KinectPlayerSelector::OnPlayerLost, this);

		if (pKinect->GetTrackedBodies().size() > 0)
		{
			Reselect();
		}
	}
}

void KinectPlayerSelector::GetTrackedArmatures(std::vector<TrackedArmature*>& armatures)
{
	auto& list = static_cast<KinectSensor*>(m_sensor.get())->GetTrackedBodies();
	armatures.clear();
	int idx = 0;
	for (auto& player : list)
		if (player.IsTracked())
			armatures.push_back(&player);
}

LeapPlayerSelector::LeapPlayerSelector(Devices::LeapSensor * pLeap, SelectionMode mode)
{
	Initialize(pLeap, mode);
}

LeapPlayerSelector::~LeapPlayerSelector()
{
}

void LeapPlayerSelector::Initialize(Devices::LeapSensor * pLeap, SelectionMode mode)
{
	this->mode = mode;
	if (pLeap)
	{
		this->m_sensor = pLeap->GetRef();
		con_tracked =
			pLeap->HandTracked += MakeEventHandler(&KinectPlayerSelector::OnPlayerTracked, this);
		con_lost =
			pLeap->HandLost += MakeEventHandler(&KinectPlayerSelector::OnPlayerLost, this);

		if (pLeap->GetTrackedHands().size() > 0)
		{
			Reselect();
		}
	}
}

void LeapPlayerSelector::GetTrackedArmatures(std::vector<TrackedArmature*>& armatures)
{
	auto& list = static_cast<LeapSensor*>(m_sensor.get())->GetTrackedHands();
	armatures.clear();
	int idx = 0;
	for (auto& player : list)
		if (player.IsTracked())
		armatures.push_back(&player);
}

float LeapPlayerSelector::Distance(const TrackedArmature & body) const
{
	return 1.0f;
}

PlayerSelectorBase::MergedTrackedArmature::MergedTrackedArmature(const std::vector<TrackedArmature*>& players)
	: TrackedArmature()
{
	assert(players.size() > 1);
	m_subarms = players;
	unique_ptr<Joint> root(new Joint(0));
	root->Name = "UnionRoot";
	int idx = 1;
	int pid = 0;
	m_id = 0;
	char pname[32];
	memset(pname, 0, sizeof(pname));

	for (auto player : players)
	{
		auto subarm = player->GetArmature().root()->clone();
		root->append_children_back(subarm);

		sprintf_s(pname, "_P%d", pid);

		for (auto& joint : subarm->nodes())
		{
			joint.ID += idx;
			joint.Name += pname;
			if (joint.parent() != root.get())
				joint.ParentID += idx;
			else 
				joint.ParentID += 1;
		}

		int sz = player->GetArmature().size();
		m_subarmidx.emplace_back(idx, sz);
		idx += sz;
		m_id += player->GetTrackId() << (8*(++pid)); //? HACK here !!!
	}
	m_signals.resize(m_subarms.size(), 0);
	m_cons.resize(m_subarms.size());
	m_frameCache.resize(idx);

	DynamicArmature darm(move(root), move(m_frameCache));
	m_armature.clone_from(darm);

	// We are not using the SetArmatureProportion here since we don't kown good default pose
	auto& frame = m_armature.default_frame();
	ComposeFrame(frame);

	BuildJointMirrorRelation(m_armature);

	m_frameCache.resize(idx);
	pid = 0;
	for (auto& subarm : m_subarms)
	{
		subarm->AddRef();
		m_cons[pid].con_frame = 
			subarm->OnFrameArrived.connect(
				bind(&MergedTrackedArmature::_CollectFrame,this,std::placeholders::_1,std::placeholders::_2));
		m_cons[pid].con_lost = 
			subarm->Lost.connect(
				bind(&MergedTrackedArmature::_CollectLost, this, std::placeholders::_1));
		++pid;
	}
}

void PlayerSelectorBase::MergedTrackedArmature::ComposeFrame(ArmatureFrameView frame)
{
	//ArmatureFrame frame(m_armature.size());
	int pid = 0;
	for (auto& player : m_subarms)
	{
		player->ReadNextFrame();
		auto& subframe = player->PeekFrame();
		FillSubarmature(frame, pid++, subframe);
	}

	SumriseRoot(frame);
}

void PlayerSelectorBase::MergedTrackedArmature::_CollectFrame(const TrackedArmature & player, const FrameType &subframe)
{
	auto& frame = m_frameCache;

	auto itr = std::find(m_subarms.begin(), m_subarms.end(), &player);
	assert(itr != m_subarms.end());
	int pid = itr - m_subarms.begin();

	FillSubarmature(m_frameCache, pid, subframe);

	//std::cout << "Frame collected : pid = "<< pid << std::endl;
	m_signals[pid] = 1;

	bool allUpdated = std::all_of(m_signals.begin(), m_signals.end(),
		[](int p) {return (bool)p;});
	if (allUpdated)
	{
		auto time = player.GetLastTrackedTime();
		m_isTracked = true;
		SumriseRoot(m_frameCache);

		// clear the signals
		std::fill(m_signals.begin(), m_signals.end(), 0);

		PushFrame(time,move(m_frameCache));
		//std::cout << "Frame pushed!" << std::endl;
		m_frameCache.resize(m_armature.size());
	}
}

void PlayerSelectorBase::MergedTrackedArmature::_CollectLost(const TrackedObject &)
{
	if (m_isTracked)
	{
		m_isTracked = false;
		this->Lost(*this);
		Disconnet();
	}
}

void PlayerSelectorBase::MergedTrackedArmature::FillSubarmature(ArmatureFrameView frame, int pid, const ArmatureFrameConstView & subframe)
{
	auto& sz = m_subarmidx[pid];
	std::copy_n(subframe.begin(), sz.second, frame.begin() + sz.first);
}

void PlayerSelectorBase::MergedTrackedArmature::SumriseRoot(ArmatureFrameView frame)
{
	int pid = 0;
	auto& rootbone = frame[0];

	rootbone.GblTranslation = Vector3::Zero;
	for (auto& joint : m_armature.root()->children())
	{
		rootbone.GblTranslation += frame[joint.ID].GblTranslation;
	}

	rootbone.GblTranslation /= m_subarms.size();
	rootbone.GblRotation = Quaternion::Identity;

	for (auto& joint : m_armature.root()->children())
		frame[joint.ID].UpdateLocalData(rootbone);
}
