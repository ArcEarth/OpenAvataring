#include "pch_bcl.h"
#include "Vicon.h"
#include <ViconReader\ViconRTClient.h>
#pragma comment(lib,"ViconReader.lib")

using namespace Causality::Devices;
using namespace Causality;
using namespace Causality::Math;

XM_ALIGNATTR
class ViconReaderImpl : public IViconClient, public ViconRTClient, public AlignedNew<XMVECTOR>
{
protected:
	void BodyDataToRigid(const BodyData & body, RigidTransform & rigid)
	{
		Quaternion q(body.QX, body.QY, body.QZ, body.QW);
		Vector3 t(body.TX, body.TY, body.TZ);

		rigid.Translation = t * m_world;
		rigid.Rotation = q * m_world.Rotation;
	}

public:
	virtual ~ViconReaderImpl()
	{
		stop();
	}

	ViconReaderImpl()
	{
		m_verbose = false;
	}

	// Inherited via IViconClient
	virtual const IsometricTransform & GetWorldTransform() override
	{
		return m_world;
	}
	virtual void SetWorldTransform(const IsometricTransform & transform) override
	{
		m_world = transform;
	}
	virtual RigidTransform GetRigid(int index) override
	{
		RigidTransform rigid;
		auto body = getBodyDataByID(index);
		BodyDataToRigid(body,rigid);
		return rigid;
	}
	virtual RigidTransform GetRigid(const string & name) override
	{
		RigidTransform rigid;
		auto body = this->getBodyData(name);
		BodyDataToRigid(body, rigid);
		return rigid;
	}
	virtual int GetRigidID(const string & name) override
	{
		return this->getBodyChannelID(name);
	}

	// Inherited via IViconClient
	virtual bool Start() override
	{
		return this->start();
	}

	virtual void Stop() override
	{
		this->stop();
	}

	virtual bool Update() override
	{
		this->update();
		if (m_verbose)
			this->showBodies(1);
		return true;
	}

	virtual void SetVerbose(bool verbose) override
	{
		m_verbose = verbose;
	}

	// Inherited via IViconClient
	virtual bool Initialize(const ParamArchive * archive) override
	{
		string sever_ip = "localhost";
		GetParam(archive, "sever_ip", sever_ip);
		GetParam(archive, "translation", m_world.Translation);
		GetParam(archive, "scale", m_world.Scale);
		Vector3 eular;
		GetParam(archive, "rotation", eular);
		m_world.Rotation = Quaternion::CreateFromYawPitchRoll(eular.y, eular.x, eular.z);
		this->init(sever_ip);
		GetParam(archive, "verbose", m_verbose);

		XMMATRIX Rot = XMMatrixIdentity();
		Rot.r[2] = Rot.r[1];
		Rot.r[1] = -g_XMIdentityR2.v;
		m_world.Rotation = XMQuaternionRotationMatrix(Rot);

		return true;
	}
	virtual bool IsStreaming() const override
	{
		return true;
	}

	virtual bool IsAsychronize() const override
	{
		return false;
	}

private:
	IsometricTransform  m_world;
	bool				m_verbose;
};

wptr<ViconReaderImpl> g_wpVicon;

sptr<IViconClient> IViconClient::GetFroCurrentView()
{
	if (!g_wpVicon.expired())
		return g_wpVicon.lock();
	return nullptr;
}

sptr<IViconClient> IViconClient::Create(const std::string &serverIP)
{
	sptr<ViconReaderImpl> pClient;
	if (g_wpVicon.expired())
	{
		pClient.reset(new ViconReaderImpl());
		g_wpVicon = pClient;
	}
	else
	{
		auto pClient = g_wpVicon.lock();
		pClient->stop();
	}

	if (!serverIP.empty())
		pClient->init(serverIP);

	return pClient;
}

IViconClient::~IViconClient()
{
}
