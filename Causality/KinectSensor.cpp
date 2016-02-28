#include "pch_bcl.h"
#include <Kinect.h>
#include "KinectSensor.h"
//#include <boost\range.hpp>
#include <iostream>
#include <chrono>
#include <wrl\event.h>


#pragma comment(lib,"Kinect20.lib")

using namespace Causality;
using namespace Causality::Devices;
using namespace Microsoft::WRL;

extern ShrinkedArmature g_PlayeerBlocks;
extern bool			 g_EnableInputFeatureLocalization;


template <class T>
inline void SafeRelease(T*& pCom)
{
	if (pCom != nullptr)
	{
		pCom->Release();
		pCom = nullptr;
	}
}

template< typename ContainerT, typename PredicateT >
void erase_if(ContainerT& items, const PredicateT& predicate) {
	for (auto it = items.begin(); it != items.end(); ) {
		if (predicate(*it)) it = items.erase(it);
		else ++it;
	}
};

JointType KinectSensor::JointsParent[JointType_Count] = {
	JointType_SpineBase, // JointType_SpineBase, Root
	JointType_SpineBase, // JointType_SpineMid
	JointType_SpineShoulder, // JointType_Neck
	JointType_Neck,  //JointType_Head
	JointType_SpineShoulder,  //JointType_ShoulderLeft
	JointType_ShoulderLeft, // JointType_ElbowLeft
	JointType_ElbowLeft, // JointType_WristLeft
	JointType_WristLeft,  // JointType_HandLeft
	JointType_SpineShoulder,  // JointType_ShoulderRight
	JointType_ShoulderRight,  // JointType_ElbowRight
	JointType_ElbowRight,  // JointType_WristRight
	JointType_WristRight,  // JointType_HandRight
	JointType_SpineBase,  // JointType_HipLeft
	JointType_HipLeft,  // JointType_KneeLeft
	JointType_KneeLeft,  // JointType_AnkleLeft
	JointType_AnkleLeft,  // JointType_FootLeft
	JointType_SpineBase,  // JointType_HipRight
	JointType_HipRight,  // JointType_KneeRight
	JointType_KneeRight,  // JointType_AnkleRight
	JointType_AnkleRight,  // JointType_FootRight
	JointType_SpineMid,  // JointType_SpineShoulder
	JointType_HandLeft, // JointType_HandTipLeft
	JointType_HandLeft,  // JointType_ThumbLeft
	JointType_HandRight,  // JointType_HandTipRight
	JointType_HandRight,  //JointType_ThumbRight
};

JointType JointsMirrows[JointType_Count] = {
	/*JointType_SpineBase*/ JointType_Count,
	/*JointType_SpineMid*/ JointType_Count,
	/*JointType_Neck*/ JointType_Count,
	/*JointType_Head*/ JointType_Count,
	/*JointType_ShoulderLeft*/ JointType_ShoulderRight,
	/*JointType_ElbowLeft*/ JointType_ElbowRight,
	/*JointType_WristLeft*/ JointType_WristRight,
	/*JointType_HandLeft*/ JointType_HandRight,
	/*JointType_ShoulderRight*/ JointType_ShoulderLeft,
	/*JointType_ElbowRight*/ JointType_ElbowLeft,
	/*JointType_WristRight*/ JointType_WristLeft,
	/*JointType_HandRight*/ JointType_HandLeft,
	/*JointType_HipLeft*/ JointType_HipRight,
	/*JointType_KneeLeft*/ JointType_KneeRight,
	/*JointType_AnkleLeft*/ JointType_AnkleRight,
	/*JointType_FootLeft*/ JointType_FootRight,
	/*JointType_HipRight*/ JointType_HipLeft,
	/*JointType_KneeRight*/ JointType_KneeLeft,
	/*JointType_AnkleRight*/ JointType_AnkleLeft,
	/*JointType_FootRight*/ JointType_FootLeft,
	/*JointType_SpineShoulder*/ JointType_SpineShoulder,
	/*JointType_HandTipLeft*/ JointType_HandTipRight,
	/*JointType_ThumbLeft*/ JointType_ThumbRight,
	/*JointType_HandTipRight*/ JointType_HandTipLeft,
	/*JointType_ThumbRight*/ JointType_ThumbLeft,
};

char* JointNames[JointType_Count] = {
	"SpineBase",
	"SpineMid",
	"Neck",
	"Head",
	"ShoulderLeft",
	"ElbowLeft",
	"WristLeft",
	"HandLeft",
	"ShoulderRight",
	"ElbowRight",
	"WristRight",
	"HandRight",
	"HipLeft",
	"KneeLeft",
	"AnkleLeft",
	"FootLeft",
	"HipRight",
	"KneeRight",
	"AnkleRight",
	"FootRight",
	"SpineShoulder",
	"HandTipLeft",
	"ThumbLeft",
	"HandTipRight",
	"ThumbRight",
};

std::unique_ptr<StaticArmature> TrackedBody::BodyArmature;

struct FilteredPlayer : public TrackedBody
{
	std::array<Vector3DynamicFilter, JointType_Count> JointFilters;
};

class KinectSensor::Impl
{


public:
	size_t											LostThreshold = 15U;
	time_t											LastFrameTime = 0;
	float											FrameRate = 30;
	bool											m_EnableMirrow = false;

	KinectSensor*									pWrapper;
	DirectX::Plane									FloorPlane;
	DirectX::Matrix4x4								LocalMatrix;

	// Current Kinect
	Microsoft::WRL::ComPtr<IKinectSensor>			m_pKinectSensor;
	Microsoft::WRL::ComPtr<ICoordinateMapper>		m_pCoordinateMapper;
	// Body reader
	Microsoft::WRL::ComPtr<IBodyFrameSource>		m_pBodyFrameSource;
	Microsoft::WRL::ComPtr<IBodyFrameReader>		m_pBodyFrameReader;
	Microsoft::WRL::Wrappers::Event					m_FrameReadyEvent;

	std::list<TrackedBody>							m_Players;
	std::unique_ptr<std::thread>					m_pThread;

	Impl()
	{

	}

	~Impl()
	{
		StopTracking();
	}
public:
	bool IsActive() const
	{
		return m_pKinectSensor;
	}

	HRESULT Initalize()
	{

		InitializeBodyArmature();
		using namespace Microsoft::WRL;
		using namespace std;
		HRESULT hr;
		// Current Kinect
		ComPtr<IKinectSensor>          pKinectSensor;
		ComPtr<ICoordinateMapper>      pCoordinateMapper;
		ComPtr<IBodyFrameReader>       pBodyFrameReader;
		ComPtr<IBodyFrameSource>	   pBodyFrameSource = NULL;

		hr = GetDefaultKinectSensor(&pKinectSensor);
		if (FAILED(hr))
		{
			return hr;
		}

		m_pKinectSensor = move(pKinectSensor);

		return hr;

		if (pKinectSensor)
		{
			// Initialize the Kinect and get coordinate mapper and the body reader

			hr = pKinectSensor->Open();

			if (SUCCEEDED(hr))
			{
				hr = pKinectSensor->get_CoordinateMapper(&pCoordinateMapper);
			}

			if (SUCCEEDED(hr))
			{
				hr = pKinectSensor->get_BodyFrameSource(&pBodyFrameSource);
			}
		}

		if (!pKinectSensor || FAILED(hr))
		{
			std::cout << "[DEBUG] Failed to intialize KINECT" << std::endl;
			//SetStatusMessage(L"No ready Kinect found!", 10000, true);
			return hr;
		}

		m_pKinectSensor = move(pKinectSensor);
		m_pBodyFrameSource = move(pBodyFrameSource);
		m_pCoordinateMapper = move(pCoordinateMapper);
		std::cout << "[DEBUG] Kinect is intialized succuessfuly!" << std::endl;
		return hr;
	}

	void InitializeBodyArmature()
	{
		if (TrackedBody::BodyArmature == nullptr)
		{
			int parents[JointType_Count];
			std::copy_n(KinectSensor::JointsParent, (int)JointType_Count, parents);
			TrackedBody::BodyArmature = std::make_unique<StaticArmature>(JointType_Count, parents, JointNames);
			auto& armature = *TrackedBody::BodyArmature;
			for (int i = 0; i < armature.size(); i++)
			{
				if (JointsMirrows[i] < JointType_Count)
				{
					armature[i]->MirrorJoint = armature[JointsMirrows[i]];
				}
			}
		}
	}

	IBodyFrameReader* BodyFrameReader() const
	{
		return m_pBodyFrameReader.Get();
	}

	ICoordinateMapper* CoordinateMapper() const
	{
		return m_pCoordinateMapper.Get();
	}

	IKinectGestureRecognizer* GestureRecongnizer() const
	{
		return nullptr;
	}

	bool StartTracking()
	{
		if (IsStarted())
			return Resume();

		WAITABLE_HANDLE hBFEvent = NULL;
		HRESULT hr;
		BOOLEAN active = FALSE;

		if (m_pKinectSensor)
		{
			// Initialize the Kinect and get coordinate mapper and the body reader

			hr = m_pKinectSensor->Open();

			if (SUCCEEDED(hr))
			{
				hr = m_pKinectSensor->get_BodyFrameSource(&m_pBodyFrameSource);
			}
		}

		if (!m_pKinectSensor || FAILED(hr))
		{
			std::cout << "[DEBUG] Failed to intialize KINECT" << std::endl;
			//SetStatusMessage(L"No ready Kinect found!", 10000, true);
			return false;
		}

		std::cout << "[DEBUG] Kinect is intialized succuessfuly!" << std::endl;

		hr = m_pBodyFrameSource->OpenReader(&m_pBodyFrameReader);

		if (FAILED(hr))
			return false;

		m_pBodyFrameSource->get_IsActive(&active);
		if (!active)
			return false;

		hr = m_pBodyFrameReader->SubscribeFrameArrived(&hBFEvent);

		if (FAILED(hr))
			return false;

		m_FrameReadyEvent.Attach(reinterpret_cast<HANDLE>(hBFEvent));

		m_pBodyFrameReader->put_IsPaused(FALSE);

		m_pThread = std::make_unique<std::thread>([this](){
			while (m_FrameReadyEvent.IsValid())
			{
				auto result = WaitForSingleObject(m_FrameReadyEvent.Get(), 0);
				switch (result)
				{
				case WAIT_ABANDONED:
					throw;
				case WAIT_FAILED:
					m_FrameReadyEvent.Detach();
					return;
				case WAIT_TIMEOUT:
					continue;
				case WAIT_OBJECT_0:
					break;
				};

				ComPtr<IBodyFrameArrivedEventArgs> args;
				ComPtr<IBodyFrameReference> frameRef;
				ComPtr<IBodyFrame> pBodyFrame;

				HRESULT hr = m_pBodyFrameReader->GetFrameArrivedEventData(reinterpret_cast<WAITABLE_HANDLE>(m_FrameReadyEvent.Get()), &args);
				if (FAILED(hr))
					continue;
				hr = args->get_FrameReference(&frameRef);
				if (FAILED(hr))
					continue;
				hr = frameRef->AcquireFrame(&pBodyFrame);
				if (FAILED(hr))
					continue;

				ProcessFrame(pBodyFrame.Get());
			}
		});
		return true;
	}

	void StopTracking()
	{
		if (!m_FrameReadyEvent.IsValid())
			return;
		if (m_pBodyFrameReader && m_FrameReadyEvent.Get() != NULL)
		{
			m_pBodyFrameReader->put_IsPaused(TRUE);
			m_pBodyFrameReader->UnsubscribeFrameArrived(reinterpret_cast<WAITABLE_HANDLE>(m_FrameReadyEvent.Get()));
			m_FrameReadyEvent.Detach();
		}
		if (m_pThread)
		{
			m_pThread->join();
			m_pThread.reset();
		}
	}

	bool IsTracking() const
	{
		BOOLEAN paused;
		m_pBodyFrameReader->get_IsPaused(&paused);
		return m_pThread != nullptr && !paused;
	}

	bool IsPaused()
	{
		BOOLEAN paused;
		m_pBodyFrameReader->get_IsPaused(&paused);
		return paused;
	}

	bool IsStarted()
	{
		return m_pThread != nullptr;
	}

	bool Pause(bool pause)
	{
		return SUCCEEDED(m_pBodyFrameReader->put_IsPaused((BOOLEAN)pause));
	}

	bool Resume()
	{
		return SUCCEEDED(m_pBodyFrameReader->put_IsPaused(FALSE));
	}

	bool HasNewFrame() const
	{
	}

	bool ProcessFrame()
	{
		ComPtr<IBodyFrame> pBodyFrame;
		HRESULT hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);

		if (FAILED(hr))
		{
			return false;
		}

		return ProcessFrame(pBodyFrame.Get());
	}

	bool ProcessFrame(IBodyFrame* pBodyFrame)
	{
		time_t nTime = 0;

		HRESULT hr = pBodyFrame->get_RelativeTime(&nTime);

		if (FAILED(hr))
		{
			return false;
		}

		//auto t = std::chrono::system_clock::from_time_t(nTime);
		//t.time_since_epoch
		if (LastFrameTime >= nTime) // Not an new frame
		{
			return false;
		}

		LastFrameTime = nTime;

		// Process experied bodies
		for (auto itr = m_Players.begin(), end = m_Players.end(); itr != end;)
		{
			if (++(itr->m_lostFrameCount) >= (int)LostThreshold)
			{
				if (itr->m_isTracked)
				{
					itr->m_isTracked = false;
					pWrapper->OnPlayerLost(*itr);
					std::cout << "Player Lost : ID = " << itr->GetTrackId() << std::endl;
				}
				if (itr->RefCount() <= 0)
					itr = m_Players.erase(itr);
				else
					++itr;
			}
			else
				++itr;
		}

		// Retrive new bodies
		IBody* ppBodies[BODY_COUNT] = { 0 };

		hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

		if (FAILED(hr))
		{
			return false;
		}

		for (int i = 0; i < _countof(ppBodies); ++i)
		{
			auto &pBody = ppBodies[i];
			if (pBody == nullptr)
				break;

			UINT64 Id;
			pBody->get_TrackingId(&Id);
			BOOLEAN isTracked;
			pBody->get_IsTracked(&isTracked);

			TrackedBody* player = nullptr;
			bool isNew = false;

			//FilteredPlayer* player = static_cast<FilteredPlayer*>(m_Players[Id]);
			auto& itr = std::find_if(m_Players.begin(), m_Players.end(), [Id](const TrackedBody& body)->bool { return body.GetTrackId() == Id;});

			if (itr != m_Players.end())
				player = &(*itr);

			if (isTracked)
			{
				// New Player tracked
				if (player == nullptr)
				{
					m_Players.emplace_back();
					player = &m_Players.back();
					player->m_pKinectSensor = this->pWrapper;
					player->m_id = Id;
					player->m_isTracked = (bool)isTracked;

					std::cout << "New Player Detected : ID = " << Id << std::endl;
					isNew = true;
				}

				//std::cout << "Player Tracked, ID = '" << Id << '\'' << std::endl;

				UpdateTrackedBodyData(player, pBody, LastFrameTime, isNew);

			}
			else if (player != nullptr) // Handel Losted player
			{
				if (player->GetLostFrameCount() > 3)
				{
					//for (auto& filter : player->JointFilters)
					//{
					//	filter.Reset();
					//}
				}

				if (!isNew && player->m_isTracked) // Maybe wait for couple of frames?
				{
					pWrapper->OnPlayerLost(*player);
					std::cout << "Player Lost : ID = " << Id << std::endl;
				}
				player->m_isTracked = false;
			}

			//SafeRelease(ppBodies[i]);
		}

		return true;
	}

	void UpdateTrackedBodyData(TrackedBody * player, IBody * pBody, time_t time, bool isNew = false)
	{
		::Joint joints[JointType_Count];
		JointOrientation oris[JointType_Count];

		const static DirectX::XMVECTORF32 g_XMNegateXZ = { -1.0f, 1.0f, -1.0f, 1.0f };

		player->m_isTracked = true;
		player->ResetLostFrameCount();

		pBody->GetJoints(ARRAYSIZE(joints), joints);
		pBody->GetJointOrientations(ARRAYSIZE(oris), oris);
		player->m_Distance = joints[0].Position.Z;

		TrackedBody::FrameType frame(JointType::JointType_Count);
		//frame.Time = std::chrono::system_clock::from_time_t(time).time_since_epoch();

		// WARNING!!! This may be ill-formed if transform contains shear or other non-rigid transformation
		DirectX::XMVECTOR junk, rotation;
		DirectX::XMMATRIX transform = LocalMatrix;

		// Flip X and Z, X is due to mirrow effect, Z is due to Kinect uses LH coordinates
		if (m_EnableMirrow)
			transform.r[0] = DirectX::XMVectorNegate(transform.r[0]);
		//transform.r[2] = DirectX::XMVectorNegate(transform.r[2]);

		// Extract the rotation to apply
		DirectX::XMMatrixDecompose(&junk, &rotation, &junk, transform);

		for (int j = 0; j < JointType_Count; j++)
		{
			//auto& filter = player->JointFilters[j];
			DirectX::XMVECTOR ep = DirectX::XMLoadFloat3(reinterpret_cast<DirectX::Vector3*>(&joints[j].Position));
			ep = DirectX::XMVector3TransformCoord(ep, transform);
			DirectX::XMVECTOR bp = DirectX::XMLoadFloat3(reinterpret_cast<DirectX::Vector3*>(&joints[JointsParent[j]].Position));
			//frame[j].GblLength = DirectX::XMVectorGetX(DirectX::XMVector3Length(ep - bp));
			//frame[j].LclLength = frame[j];

			//? Why not filtering?
			//! We are tracking gestures here, keep the raw data is better!
			frame[j].GblTranslation = ep; // filter.Apply(ep);
			ep = DirectX::XMLoadFloat4(reinterpret_cast<DirectX::Quaternion*>(&oris[j].Orientation));

			ep *= DirectX::g_XMNegateX;
			//ep *= g_XMNegateXZ.v;

			frame[j].GblRotation = DirectX::XMQuaternionMultiply(ep, rotation);
		}

		auto& armature = *TrackedBody::BodyArmature;

		for (auto jId : armature.joint_indices())
		{
			// Update 'intermediate" joints if not presented in Kinect
			auto pId = armature[jId]->ParentID;
			if (jId > JointType_Count)
			{
				frame[jId].UpdateGlobalData(frame[pId]);
			}
			DirectX::XMVECTOR q = XMLoadA(frame[jId].GblRotation);
			if (DirectX::XMVector4Equal(q, DirectX::XMVectorZero()))
			{
				if (pId != -1)
					q = XMLoadA(frame[pId].GblRotation);
				else
					q = DirectX::XMQuaternionIdentity();
				XMStoreA(frame[jId].GblRotation,q);
			}

		}

		FrameRebuildLocal(armature, frame);

		if (isNew)
		{
			player->SetArmatureProportion(frame);
			std::cout << "New Player Armature Proportion Captured" << std::endl;
		}

		player->PushFrame(time, std::move(frame));

		if (isNew)
		{
			pWrapper->OnPlayerTracked(*player);
			std::cout << "New Player Tracked : ID = " << player->GetTrackId() << std::endl;
			//player->FireFrameArrivedForLatest();
		}
		else
		{

			//player->OnFrameArrived(*player, frame);
			HandState state;
			pBody->get_HandLeftState(&state);
			if (state != player->m_HandStates[0])
			{
				player->m_HandStates[0] = state;
				//player->OnHandStateChanged(*player, HandType_LEFT, state);
			}
			pBody->get_HandRightState(&state);
			if (state != player->m_HandStates[1])
			{
				player->m_HandStates[1] = state;
				//player->OnHandStateChanged(*player, HandType_RIGHT, state);
			}
		}
	}

	//static void FillFrameWithIBody(Kinematics::ArmatureFrameView frame, IBody* pBody)
	//{
	//	Joint joints[JointType_Count];
	//	JointOrientation oris[JointType_Count];
	//	pBody->GetJoints(ARRAYSIZE(joints), joints);
	//	pBody->GetJointOrientations(ARRAYSIZE(oris), oris);
	//	oris[0].Orientation
	//};

	//auto TrackedPlayers() const
	//{
	//	using namespace boost;
	//	using namespace boost::adaptors;
	//	return m_Players | map_values;
	//}
};

std::weak_ptr<KinectSensor> KinectSensor::wpCurrentDevice;

std::shared_ptr<KinectSensor> Devices::KinectSensor::GetForCurrentView()
{
	if (wpCurrentDevice.expired())
	{
		auto pDevice = CreateDefault();
		wpCurrentDevice = pDevice;
		return pDevice;
	}
	else
	{
		return wpCurrentDevice.lock();
	}
}

KinectSensor::~KinectSensor()
{
	Stop();
}

bool KinectSensor::IsConnected() const
{
	return pImpl->IsActive();
}

void KinectSensor::SetDeviceCoordinate(const DirectX::Matrix4x4 & mat)
{
	pImpl->LocalMatrix = mat;
}

void KinectSensor::EnableMirrowing(bool enable_mirrow)
{
	pImpl->m_EnableMirrow = enable_mirrow;
}

bool Devices::KinectSensor::IsMirrowingEnable() const
{
	return pImpl->m_EnableMirrow;
}

const DirectX::Matrix4x4 & KinectSensor::GetDeviceCoordinate()
{
	return pImpl->LocalMatrix;
}

void KinectSensor::ProcessFrame()
{
	pImpl->ProcessFrame();
}

bool KinectSensor::Start()
{
	return pImpl->StartTracking();
}

void KinectSensor::Stop()
{
	pImpl->StopTracking();
}

bool KinectSensor::Pause()
{
	return pImpl->Pause(TRUE);
}

bool KinectSensor::Resume()
{
	return pImpl->Resume();
}

bool KinectSensor::Initialize(const ParamArchive * archive)
{
	return true;
}

bool KinectSensor::Update()
{
	return false;
}

bool KinectSensor::IsAsychronize() const
{
	return true;
}

bool KinectSensor::IsStreaming() const
{
	return pImpl->IsActive();
}

const std::list<TrackedBody>& KinectSensor::GetTrackedBodies() const
{
	return pImpl->m_Players;
}

std::list<TrackedBody>& KinectSensor::GetTrackedBodies()
{
	return pImpl->m_Players;
}

// Static Constructors!!!

std::shared_ptr<KinectSensor> KinectSensor::CreateDefault()
{
	std::shared_ptr<KinectSensor> pKinect = std::make_shared<KinectSensor>();
	if (SUCCEEDED(pKinect->pImpl->Initalize()))
		return pKinect;
	else
		return nullptr;
}

KinectSensor::KinectSensor()
	:pImpl(new KinectSensor::Impl)
{
	pImpl->pWrapper = this;
}

TrackedBody::TrackedBody(size_t bufferSize)
	: TrackedArmature(*BodyArmature, bufferSize)
{
}

//void TrackedBody::PushFrame(FrameType && frame)
//{
//	m_FrameBuffer.Push(std::move(frame));
//	FireFrameArrivedForLatest();
//}
//
//bool TrackedBody::ReadLatestFrame()
//{
//	return m_FrameBuffer.MoveToLatest();
//}
//
//bool TrackedBody::ReadNextFrame()
//{
//	return m_FrameBuffer.MoveNext();
//}
//
//const TrackedBody::FrameType& TrackedBody::PeekFrame() const
//{
//	return *m_FrameBuffer.GetCurrent();
//}
//
//const IArmature & TrackedBody::GetArmature() const {
//	return m_armature; 
//}
//
//bool TrackedBody::IsAvailable() const
//{
//	return IsTracked();
//}
//
//float TrackedBody::DistanceToSensor() const
//{
//	return m_Distance;
//}