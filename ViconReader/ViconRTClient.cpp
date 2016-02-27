#define _USE_MATH_DEFINES
#include "ViconRTClient.h"
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <winsock2.h>
#include "TimingHelper.h"
#pragma warning(push)
#pragma warning(disable,4244)

#pragma comment(lib,"ws2_32.lib")

const ViconRTClient::Ref ViconRTClient::Empty = ViconRTClient::Ref(nullptr);

const MarkerData ViconRTClient::EmptyMarkerData;

ViconRTClient::Ref ViconRTClient::create(const std::string &server_ip, const std::string &prefix) {
	return std::make_shared<ViconRTClient>(server_ip, prefix);
	//allocate_shared<>
}

ViconRTClient::Ref ViconRTClient::getPtr() {
	return shared_from_this();
}

ViconRTClient::ViconRTClient(const std::string &server_ip, const std::string &prefix) :m_server_ip(server_ip), m_stop(true), m_prefix(prefix) {
	//pthread_barrier_init(&m_barrier,NULL,2);
	m_recording = false;
	m_count = 0;

	FrameChannel = -1;
	TimeVChannel = -1;
	TimeRChannel = -1;
	TimeHChannel = -1;
	TimeMChannel = -1;
	TimeSChannel = -1;
	TimeMSChannel = -1;
	TimeFChannel = -1;
	TimeOFFChannel = -1;

	m_framerate = 0.0f;
	m_timecodeValidity = 0.0f;
	m_timecodeRate = 0.0f;
	m_timecodeHours = 0.0f;
	m_timecodeMinutes = 0.0f;
	m_timecodeSeconds = 0.0f;
	m_timecodeMSeconds = 0.0f;
	m_timecodeFrame = 0.0f;
	m_timecodeOffset = 0.0f;

	m_initSkeleton = false;

	Eigen::Affine3f matrix;
	matrix = Eigen::Scaling(0.001f);
};

ViconRTClient::~ViconRTClient(void) {
	if (!m_stop) stop();
}

bool ViconRTClient::start() {
	if (!m_stop) stop();
	m_thread = std::thread(&ViconRTClient::runStatic, this);
	//int rc = pthread_create(&m_thread, NULL,&ViconRTClient::runStatic,this);
	//if (rc){
	//	printf("ERROR; return code from pthread_create() is %d\n", rc);
	//	return false;
	//} else {
	//	pthread_barrier_wait(&m_barrier);
	//}
	return true;
}

void ViconRTClient::run() {
	//pthread_barrier_wait(&m_barrier);
	if (!initialize(m_server_ip.c_str())) {
		getChannelInfo();
		getFrame();
		//initSkeleton();
		showMarkers(VERBOSE);
		showBodies(VERBOSE);
		//int IDRobot=getBodyChannelID("Robot:Robot");
		m_stop = false;
		while (!m_stop) {
			double timestamp_s = getFrame();
			//showBodies(VERBOSE);
			//showMarkers(VERBOSE);
			recordFileData();
			//const BodyData data=m_viconClient.getBodyDataByID(IDRobot);
			//data.TX,data.TY/10.0,data.TZ;
		}
		close();
	}
}

void *ViconRTClient::runStatic(void *val) {
	if (val) {
		((ViconRTClient *)val)->run();
		//delete val;
	}
	//pthread_exit(NULL);
	return 0;
}

void ViconRTClient::stop() {
	m_stop = true;
}


void ViconRTClient::init(const std::string &server_ip, const std::string &prefix) {
	m_prefix = prefix;
	m_server_ip = server_ip;
	m_stop = true;
	m_recording = false;
	m_count = 0;
}

void ViconRTClient::setPrefix(const std::string &prefix) {
	m_prefix = prefix;
}


void ViconRTClient::showMarkers(int toShow) const {


	std::vector< MarkerChannel >::const_iterator iMarker;
	std::vector< MarkerData >::const_iterator iMarkerData;

	//#ifdef VC_VERBOSE
	std::cout << "ViconRTClient: Showing Markers Data" << std::endl;
	//#endif VC_VERBOSE

	for (iMarker = MarkerChannels.begin(), iMarkerData = markerPositions.begin();
	iMarker != MarkerChannels.end();
		iMarker++, iMarkerData++) {

		if (toShow == BRIEF) {
			std::cout << iMarker->Name << std::endl;
		}
		else {
			std::cout << iMarker->Name << " ";
			std::cout << iMarkerData->X << ", " << iMarkerData->Y << ", " << iMarkerData->Z << std::endl;
		}
	}
}

void ViconRTClient::showBodies(int toShow) const {


	std::vector< BodyChannel >::const_iterator iBody;
	std::vector< BodyData >::const_iterator iBodyData;

	if (bodyPositions.size() < BodyChannels.size()) {
		std::cout << "ViconRTClient: No data yet" << std::endl;
		return;
	}
	std::cout << "ViconRTClient: Showing Body Data" << std::endl;

	for (iBody = BodyChannels.begin(),
		iBodyData = bodyPositions.begin();
		iBody != BodyChannels.end(); ++iBody)
	{

		if (toShow == BRIEF) {
			std::cout << iBody->Name << std::endl;
		}
		else {
			std::cout << iBody->Name << " pos(";
			if (iBodyData != bodyPositions.end())
			{
				std::cout << iBodyData->TX << ", " << iBodyData->TY << ", " << iBodyData->TZ << ") ";
				std::cout << " rot(" << iBodyData->EulerX << ", " << iBodyData->EulerY << ", " << iBodyData->EulerZ << ") " << std::endl;
				++iBodyData;
			}
			else
			{
				std::cout << "no data)" << std::endl;
			}
		}
	}
}



int ViconRTClient::initialize(const std::string &server_ip_addr) {


#ifdef VC_VERBOSE
	std::cout << "ViconRTClient: Initializing Winsock" << std::endl;
#endif VC_VERBOSE

	WORD wVersionRequested;
	WSADATA wsaData;
	wVersionRequested = MAKEWORD(2, 0);
	if (WSAStartup(wVersionRequested, &wsaData) != 0)
	{
		std::cout << "ViconRTClient: Socket Initialization Error" << std::endl;
		return 1;
	}

	// Create Socket

	SocketHandle = INVALID_SOCKET;

	struct protoent*	pProtocolInfoEntry;
	char*	protocol;
	int		type;

	protocol = "tcp";
	type = SOCK_STREAM;

	pProtocolInfoEntry = getprotobyname(protocol);
	assert(pProtocolInfoEntry);

	if (pProtocolInfoEntry)
		SocketHandle = socket(PF_INET, type, pProtocolInfoEntry->p_proto);

	if (SocketHandle == INVALID_SOCKET)
	{
		std::cout << "ViconRTClient: Socket Creation Error" << std::endl;
		return 1;
	}


#ifdef VC_VERBOSE
	std::cout << "ViconRTClient: Initializing Socket Connection to " << server_ip_addr << std::endl;
#endif VC_VERBOSE

	//	Find Endpoint

	struct hostent*		pHostInfoEntry;
	struct sockaddr_in	Endpoint;

	static const int port = 800;

	memset(&Endpoint, 0, sizeof(Endpoint));
	Endpoint.sin_family = AF_INET;
	Endpoint.sin_port = htons(port);

	pHostInfoEntry = gethostbyname(server_ip_addr.c_str());

	if (pHostInfoEntry)
		memcpy(&Endpoint.sin_addr, pHostInfoEntry->h_addr,
			pHostInfoEntry->h_length);
	else
		Endpoint.sin_addr.s_addr = inet_addr(server_ip_addr.c_str());

	if (Endpoint.sin_addr.s_addr == INADDR_NONE)
	{
		std::cout << "ViconRTClient: Bad Address" << std::endl;
		return 1;
	}

	//	Create Socket

	int result = connect(SocketHandle, (struct sockaddr*) & Endpoint, sizeof(Endpoint));

	if (result == SOCKET_ERROR)
	{
		std::cout << "ViconRTClient: Failed to create Socket" << std::endl;
		int e = WSAGetLastError();
		return 1;
	}

	return 0;

}


int ViconRTClient::close(void) {


	if (closesocket(SocketHandle) == SOCKET_ERROR)
	{
		std::cout << "ViconRTClient: Failed to close Socket" << std::endl;
		return 1;
	}

	WSACleanup();
#ifdef VC_VERBOSE
	std::cout << "ViconRTClient: De-Initialized WinSock" << std::endl;
#endif VC_VERBOSE

	return 0;

}


//-----------------------------------------------------------------------------
//	The recv call may return with a half-full buffer.
//	revieve keeps going until the buffer is actually full.
bool receive(SOCKET Socket, char * pBuffer, int BufferSize) {
	char * p = pBuffer;
	char * e = pBuffer + BufferSize;
	int result;
	while (p != e) {
		result = recv(Socket, p, e - p, 0);
		if (result == SOCKET_ERROR)
			return false;
		p += result;
	}
	return true;
}

//	There are also some helpers to make the code a little less ugly.
bool receive(SOCKET Socket, long int & Val) {
	return receive(Socket, (char*)& Val, sizeof(Val));
}

bool receive(SOCKET Socket, unsigned long int & Val) {
	return receive(Socket, (char*)& Val, sizeof(Val));
}

bool receive(SOCKET Socket, double & Val) {
	return receive(Socket, (char*)& Val, sizeof(Val));
}

void ViconRTClient::getChannelInfo(void) {

#ifdef VC_VERBOSE
	std::cout << "ViconRTClient: Getting Channel Info" << std::endl;
#endif VC_VERBOSE
	try
	{

		//- Get Info - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		//	Request the channel information

		pBuff = buff;

		*((long int *)pBuff) = ClientCodes::EInfo;
		pBuff += sizeof(long int);
		*((long int *)pBuff) = ClientCodes::ERequest;
		pBuff += sizeof(long int);

		if (send(SocketHandle, buff, pBuff - buff, 0) == SOCKET_ERROR)
			throw std::string("Error Requesting");

		long int packet;
		long int type;


		if (!receive(SocketHandle, packet))
			throw std::string("Error Recieving");



		if (!receive(SocketHandle, type))
			throw std::string("Error Recieving");

		if (type != ClientCodes::EReply)
			throw std::string("Bad Packet");

		if (packet != ClientCodes::EInfo)
			throw std::string("Bad Reply Type");

		if (!receive(SocketHandle, size))
			throw std::string();

		info.resize(size);

		std::vector< std::string >::iterator iInfo;

		for (iInfo = info.begin(); iInfo != info.end(); iInfo++)
		{
			long int s;
			char c[255];
			char * p = c;

			if (!receive(SocketHandle, s))
				throw std::string();

			if (!receive(SocketHandle, c, s))
				throw std::string();

			p += s;

			*p = 0;

			*iInfo = std::string(c);
		}

		//- Parse Info - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		//	The info packets now contain the channel names.
		//	Identify the channels with the various dof's.

		//std::cout<<"channel info :"<<std::endl;

		for (iInfo = info.begin(); iInfo != info.end(); iInfo++)
		{
			//	Extract the channel type

			int openBrace = iInfo->find('<');

			if (openBrace == iInfo->npos)
				throw std::string("Bad Channel Id");

			int closeBrace = iInfo->find('>');

			if (closeBrace == iInfo->npos)
				throw std::string("Bad Channel Id");

			closeBrace++;

			std::string Type = iInfo->substr(openBrace, closeBrace - openBrace);

			//std::cout<<Type<<" ";

			//	Extract the Name

			std::string Name = iInfo->substr(0, openBrace);

			int space = Name.rfind(' ');

			if (space != Name.npos)
				Name.resize(space);

			std::vector< MarkerChannel >::iterator iMarker;
			std::vector< BodyChannel >::iterator iBody;
			std::vector< std::string >::const_iterator iTypes;

			iMarker = std::find(MarkerChannels.begin(),
				MarkerChannels.end(), Name);

			iBody = std::find(BodyChannels.begin(), BodyChannels.end(), Name);

			if (iMarker != MarkerChannels.end())
			{
				//	The channel is for a marker we already have.
				iTypes = std::find(ClientCodes::MarkerTokens.begin(), ClientCodes::MarkerTokens.end(), Type);
				if (iTypes != ClientCodes::MarkerTokens.end())
					iMarker->operator[](iTypes - ClientCodes::MarkerTokens.begin()) = iInfo - info.begin();
			}
			else
				if (iBody != BodyChannels.end())
				{
					//	The channel is for a body we already have.
					iTypes = std::find(ClientCodes::BodyTokens.begin(), ClientCodes::BodyTokens.end(), Type);
					if (iTypes != ClientCodes::BodyTokens.end())
						iBody->operator[](iTypes - ClientCodes::BodyTokens.begin()) = iInfo - info.begin();
				}
				else
					if ((iTypes = std::find(ClientCodes::MarkerTokens.begin(), ClientCodes::MarkerTokens.end(), Type))
						!= ClientCodes::MarkerTokens.end())
					{
						//	Its a new marker.
						MarkerChannels.push_back(MarkerChannel(Name));
						MarkerChannels.back()[iTypes - ClientCodes::MarkerTokens.begin()] = iInfo - info.begin();
					}
					else
						if ((iTypes = std::find(ClientCodes::BodyTokens.begin(), ClientCodes::BodyTokens.end(), Type))
							!= ClientCodes::BodyTokens.end())
						{
							//	Its a new body.
							BodyChannels.push_back(BodyChannel(Name));
							BodyChannels.back()[iTypes - ClientCodes::BodyTokens.begin()] = iInfo - info.begin();
						}
						else
							if (Type == "<F>") {
								FrameChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-V>") {
								TimeVChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-R>") {
								TimeRChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-H>") {
								TimeHChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-M>") {
								TimeMChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-S>") {
								TimeSChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-MS>") {
								TimeMSChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-F>") {
								TimeFChannel = iInfo - info.begin();
							}
							else if (Type == "<TC-OFF>") {
								TimeOFFChannel = iInfo - info.begin();
							}
							else {
								//	It could be a new channel type.
							}
		}

		markerPositions.resize(MarkerChannels.size());
		std::vector< MarkerData >::iterator iMarker;
		bodyPositions.resize(BodyChannels.size());

		std::vector< BodyChannel >::iterator iBody;
		std::vector< BodyData >::iterator iBodyData;
		for (iBody = BodyChannels.begin(),
			iBodyData = bodyPositions.begin();
			iBody != BodyChannels.end(); iBody++, iBodyData++) {
			Bones.push_back(BoneData(&(*iBody), &(*iBodyData)));
		}
	}

	catch (const std::string & rMsg)
	{
		std::cout << "ViconRTClient: ERROR in getChannelInfo --> ";
		if (rMsg.empty())
			std::cout << "Unknown Error" << std::endl;
		else
			std::cout << rMsg.c_str() << std::endl;
	}
}


double ViconRTClient::getFrame(void) {

	//	A connection with the Vicon Realtime system is now open.
	//	The following section implements the new Computer Graphics Client interface.

#ifdef VC_VERBOSE
	std::cout << "ViconRTClient: Getting Frame" << std::endl;
#endif VC_VERBOSE
	double timestamp = 0;

	try
	{
		//- Get Data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
		//	Get the data using the request/reply protocol.
		std::vector< double > data;
		data.resize(info.size());

		pBuff = buff;

		*((long int *)pBuff) = ClientCodes::EData;
		pBuff += sizeof(long int);
		*((long int *)pBuff) = ClientCodes::ERequest;
		pBuff += sizeof(long int);

		if (send(SocketHandle, buff, pBuff - buff, 0) == SOCKET_ERROR)
			throw std::string("Error Requesting");

		long int packet;
		long int type;

		//	Get and check the packet header.

		// timer loop to avoid deadlocking on receive
		//std::cout << "waiting " << dwStart << " ticks" << std::endl;
		DWORD dwStart = GetTickCount();
		while (GetTickCount() - dwStart < 20);

		if (!receive(SocketHandle, packet))
			throw std::string("Error Recieving");

		if (!receive(SocketHandle, type))
			throw std::string("Error Recieving");

		if (type != ClientCodes::EReply)
			throw std::string("Bad Packet");

		if (packet != ClientCodes::EData)
			throw std::string("Bad Reply Type");

		if (!receive(SocketHandle, size))
			throw std::string();

		if (size != info.size())
			throw std::string("Bad Data Packet");

		//	Get the data.

		std::vector< double >::iterator iData;

		for (iData = data.begin(); iData != data.end(); iData++)
		{
			if (!receive(SocketHandle, *iData))
				throw std::string();
		}

		//- Look Up Channels - - - - - - - - - - - - - - - - - - - - - - - 
		if (FrameChannel >= 0) m_framerate = data[FrameChannel];
		if (TimeVChannel >= 0) m_timecodeValidity = data[TimeVChannel];
		if (TimeRChannel >= 0) m_timecodeRate = data[TimeRChannel];
		if (TimeHChannel >= 0) m_timecodeHours = data[TimeHChannel];
		if (TimeMChannel >= 0) m_timecodeMinutes = data[TimeMChannel];
		if (TimeSChannel >= 0) m_timecodeSeconds = data[TimeSChannel];
		if (TimeMSChannel >= 0) m_timecodeMSeconds = data[TimeMSChannel];
		if (TimeFChannel >= 0) m_timecodeFrame = data[TimeFChannel];
		if (TimeOFFChannel >= 0) m_timecodeOffset = data[TimeOFFChannel];
		timestamp = /*TimingHelper::getCurrentTime();*/data[FrameChannel] / 120.0f;

		//printf("TIME: %.0f:%.0f:%.0f:%.0f \n",data[TimeHChannel],data[TimeMChannel],data[TimeSChannel],data[TimeMSChannel]);
		//	Get the channels corresponding to the markers.
		//	Y is up
		//	The values are in millimeters

		std::vector< MarkerChannel >::iterator iMarker;
		std::vector< MarkerData >::iterator iMarkerData;
		for (iMarker = MarkerChannels.begin(),
			iMarkerData = markerPositions.begin();
			iMarker != MarkerChannels.end(); iMarker++, iMarkerData++) {
			iMarkerData->X = data[iMarker->X];
			iMarkerData->Y = data[iMarker->Y];
			iMarkerData->Z = data[iMarker->Z];
			if (data[iMarker->O] > 0.5)
				iMarkerData->Visible = false;
			else
				iMarkerData->Visible = true;
		}

		//	Get the channels corresponding to the bodies.
		//=================================================================
		//	The bodies are in global space
		//	The world is Z-up
		//	The translational values are in millimeters
		//	The rotational values are in radians
		//=================================================================

		std::vector< BodyChannel >::iterator iBody;
		std::vector< BodyData >::iterator iBodyData;
		std::vector< BoneData >::iterator iBoneData;

		for (iBody = BodyChannels.begin(),
			iBodyData = bodyPositions.begin(),
			iBoneData = Bones.begin();
			iBody != BodyChannels.end(); iBody++, iBodyData++, iBoneData++) {
			iBodyData->TX = data[iBody->TX];
			iBodyData->TY = data[iBody->TY];
			iBodyData->TZ = data[iBody->TZ];

			/*
			iBodyData->baX = data[iBody->baX];
			iBodyData->baY = data[iBody->baY];
			iBodyData->baZ = data[iBody->baZ];
			*/

			iBodyData->btX = data[iBody->btX];
			iBodyData->btY = data[iBody->btY];
			iBodyData->btZ = data[iBody->btZ];

			//	The channel data is in the angle-axis form.
			//	The following converts this to a quaternion.
			//=============================================================
			//	An angle-axis is vector, the direction of which is the axis
			//	of rotation and the length of which is the amount of 
			//	rotation in radians. 
			//=============================================================

			double len, tmp;

			len = sqrt(data[iBody->RX] * data[iBody->RX] +
				data[iBody->RY] * data[iBody->RY] +
				data[iBody->RZ] * data[iBody->RZ]);

			iBodyData->QW = cos(len / 2.0);
			tmp = sin(len / 2.0);
			if (len < 1e-10) {
				iBodyData->QX = data[iBody->RX];
				iBodyData->QY = data[iBody->RY];
				iBodyData->QZ = data[iBody->RZ];
			}
			else {
				iBodyData->QX = data[iBody->RX] * tmp / len;
				iBodyData->QY = data[iBody->RY] * tmp / len;
				iBodyData->QZ = data[iBody->RZ] * tmp / len;
			}

			//	The following converts angle-axis to a rotation matrix.

			double c = 0.0, s = 0.0, x = 0.0, y = 0.0, z = 0.0;

			if (len < 1e-15) {
				iBodyData->GlobalRotation[0][0] = iBodyData->GlobalRotation[1][1] = iBodyData->GlobalRotation[2][2] = 1.0;
				iBodyData->GlobalRotation[0][1] = iBodyData->GlobalRotation[0][2] = iBodyData->GlobalRotation[1][0] =
					iBodyData->GlobalRotation[1][2] = iBodyData->GlobalRotation[2][0] = iBodyData->GlobalRotation[2][1] = 0.0;
			}
			else {
				x = data[iBody->RX] / len;
				y = data[iBody->RY] / len;
				z = data[iBody->RZ] / len;

				c = cos(len);
				s = sin(len);

				iBodyData->GlobalRotation[0][0] = c + (1 - c)*x*x;
				iBodyData->GlobalRotation[0][1] = (1 - c)*x*y + s*(-z);
				iBodyData->GlobalRotation[0][2] = (1 - c)*x*z + s*y;
				iBodyData->GlobalRotation[1][0] = (1 - c)*y*x + s*z;
				iBodyData->GlobalRotation[1][1] = c + (1 - c)*y*y;
				iBodyData->GlobalRotation[1][2] = (1 - c)*y*z + s*(-x);
				iBodyData->GlobalRotation[2][0] = (1 - c)*z*x + s*(-y);
				iBodyData->GlobalRotation[2][1] = (1 - c)*z*y + s*x;
				iBodyData->GlobalRotation[2][2] = c + (1 - c)*z*z;
			}

			// now convert rotation matrix to nasty Euler angles (yuk)
			// you could convert direct from angle-axis to Euler if you wish

			//	'Look out for angle-flips, Paul...'
			//  Algorithm: GraphicsGems II - Matrix Techniques VII.1 p 320
			assert(fabs(iBodyData->GlobalRotation[0][2]) <= 1);
			iBodyData->EulerY = asin(-iBodyData->GlobalRotation[2][0]);

			if (fabs(cos(y)) >
				std::numeric_limits<double>::epsilon()) 	// cos(y) != 0 Gimbal-Lock
			{
				iBodyData->EulerX = atan2(iBodyData->GlobalRotation[2][1], iBodyData->GlobalRotation[2][2]);
				iBodyData->EulerZ = atan2(iBodyData->GlobalRotation[1][0], iBodyData->GlobalRotation[0][0]);
			}
			else {
				iBodyData->EulerZ = 0;
				iBodyData->EulerX = atan2(iBodyData->GlobalRotation[0][1], iBodyData->GlobalRotation[1][1]);
			}

			//Local
			len = sqrt(data[iBody->baX] * data[iBody->baX] +
				data[iBody->baY] * data[iBody->baY] +
				data[iBody->baZ] * data[iBody->baZ]);

			iBodyData->bqW = cos(len / 2.0);
			tmp = sin(len / 2.0);
			if (len < 1e-10) {
				iBodyData->bqX = data[iBody->baX];
				iBodyData->bqY = data[iBody->baY];
				iBodyData->bqZ = data[iBody->baZ];
			}
			else {
				iBodyData->bqX = data[iBody->baX] * tmp / len;
				iBodyData->bqY = data[iBody->baY] * tmp / len;
				iBodyData->bqZ = data[iBody->baZ] * tmp / len;
			}

			c = 0.0; s = 0.0; x = 0.0; y = 0.0; z = 0.0;

			if (len < 1e-15) {
				iBodyData->bGlobalRotation[0][0] = iBodyData->bGlobalRotation[1][1] = iBodyData->bGlobalRotation[2][2] = 1.0;
				iBodyData->bGlobalRotation[0][1] = iBodyData->bGlobalRotation[0][2] = iBodyData->bGlobalRotation[1][0] =
					iBodyData->bGlobalRotation[1][2] = iBodyData->bGlobalRotation[2][0] = iBodyData->bGlobalRotation[2][1] = 0.0;
			}
			else {
				x = data[iBody->baX] / len;
				y = data[iBody->baY] / len;
				z = data[iBody->baZ] / len;

				c = cos(len);
				s = sin(len);

				iBodyData->bGlobalRotation[0][0] = c + (1 - c)*x*x;
				iBodyData->bGlobalRotation[0][1] = (1 - c)*x*y + s*(-z);
				iBodyData->bGlobalRotation[0][2] = (1 - c)*x*z + s*y;
				iBodyData->bGlobalRotation[1][0] = (1 - c)*y*x + s*z;
				iBodyData->bGlobalRotation[1][1] = c + (1 - c)*y*y;
				iBodyData->bGlobalRotation[1][2] = (1 - c)*y*z + s*(-x);
				iBodyData->bGlobalRotation[2][0] = (1 - c)*z*x + s*(-y);
				iBodyData->bGlobalRotation[2][1] = (1 - c)*z*y + s*x;
				iBodyData->bGlobalRotation[2][2] = c + (1 - c)*z*z;
			}

			// now convert rotation matrix to nasty Euler angles (yuk)
			// you could convert direct from angle-axis to Euler if you wish

			//	'Look out for angle-flips, Paul...'
			//  Algorithm: GraphicsGems II - Matrix Techniques VII.1 p 320
			assert(fabs(iBodyData->bGlobalRotation[0][2]) <= 1);
			iBodyData->beulerY = asin(-iBodyData->bGlobalRotation[2][0]);

			if (fabs(cos(y)) >
				std::numeric_limits<double>::epsilon()) 	// cos(y) != 0 Gimbal-Lock
			{
				iBodyData->beulerX = atan2(iBodyData->bGlobalRotation[2][1], iBodyData->bGlobalRotation[2][2]);
				iBodyData->beulerZ = atan2(iBodyData->bGlobalRotation[1][0], iBodyData->bGlobalRotation[0][0]);
			}
			else {
				iBodyData->beulerZ = 0;
				iBodyData->beulerX = atan2(iBodyData->bGlobalRotation[0][1], iBodyData->bGlobalRotation[1][1]);
			}
			iBoneData->toWorld = Eigen::Translation3d(iBodyData->TX, iBodyData->TY, iBodyData->TZ)*Eigen::Quaterniond(iBodyData->QW, iBodyData->QX, iBodyData->QY, iBodyData->QZ);
			iBoneData->toLocal = Eigen::Translation3d(iBodyData->btX, iBodyData->btY, iBodyData->btZ)*Eigen::Quaterniond(iBodyData->bqW, iBodyData->bqX, iBodyData->bqY, iBodyData->bqZ);
		}
		//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		//  The marker and body data now resides in the arrays 
		//	markerPositions & bodyPositions.
#ifdef VC_VERBOSE
		std::cout << "ViconRTClient: Frame: " << timestamp << std::endl;
#endif VC_VERBOSE
	}
	catch (const std::string & rMsg) {
		std::cout << "ViconRTClient: ERROR in getFrame --> ";
		if (rMsg.empty())
			std::cout << "Unknown Error" << std::endl;
		else
			std::cout << rMsg.c_str() << std::endl;
	}

	return timestamp;
}

int ViconRTClient::getBodyChannelID(const std::string &name) const {
	auto iBody1 = BodyChannels.begin();
	auto iBody = std::find_if(BodyChannels.begin(), BodyChannels.end(),
		[&name](const auto& body) {
		return body.Name == name;}
	);
	int id = iBody - iBody1;
	if (iBody != BodyChannels.end() && bodyPositions.size() > id)
		return id;

	return -1;
}

const BodyData &ViconRTClient::getBodyDataByID(int i) const {
	return bodyPositions[i];
}

const MarkerData &ViconRTClient::getMarkerData(const std::string &name) const {
	std::vector<MarkerChannel>::const_iterator iMarker;
	std::vector<MarkerData>::const_iterator iMarkerData;

	for (iMarker = MarkerChannels.begin(), iMarkerData = markerPositions.begin();iMarker != MarkerChannels.end();iMarker++, iMarkerData++)
		if (iMarker->Name == name) return *iMarkerData;

	return EmptyMarkerData;
}

BodyData ViconRTClient::getBodyData(std::string name) const {


	auto iBody1 = BodyChannels.begin();
	//std::cout << "Begin: " << iBody1 << std::endl;

	auto iBody = std::find_if(BodyChannels.begin(), BodyChannels.end(),
		[&name](const auto& body) {
		return body.Name == name;}
	);	//std::cout << "Found: " << iBody << std::endl;

	int i = iBody - iBody1;

	//std::cout << "Found: " << iBody - iBody1 << std::endl;

	if (iBody != BodyChannels.end()) {
		//std::cout << iBody->Name << " " << iBody << std::endl;
		std::cout << BodyChannels[i].Name;
		printf(" (%4.0f, %4.0f, %4.0f)\n", bodyPositions[i].TX, bodyPositions[i].TY, bodyPositions[i].TZ);
		//std::cout << " (" << bodyPositions[i].TX << ", " << bodyPositions[i].TY << ", " <<	bodyPositions[i].TZ << ") ";

	}
	else {
		std::cout << "Body " << name << " not found." << std::endl;
		//return (BodyData)1;
	}

	const int WHICH = 2;
	//std::cout << BodyChannels[WHICH].Name << printf("(%4.0f, %4.0f, %4.0f)",bodyPositions[WHICH].TX, bodyPositions[WHICH].TY, bodyPositions[WHICH].TZ) << std::endl;
	//std::cout << BodyData[8]->TX << ", " << BodyData[8]->TY << ", " <<	BodyData[8]->TZ << ") ";
	//std::cout << " rot(" << BodyData[8]->EulerX << ", " << BodyData[8]->EulerY << ", " <<	BodyData[8]->EulerZ << ") " << std::endl;
	//std::cout << printf("(%4.0f, %4.0f, %4.0f)", BodyData[2].TX, BodyData[2].TY, BodyData[2].TZ) << std::endl; 
	//return 0;
	return bodyPositions[i];
}

void ViconRTClient::record(int numframes) {
	if (!m_recording) {
		m_count = numframes;
		openFile();
		m_recording = true;
	}
}

void ViconRTClient::startRecording() {
	if (m_recording) stopRecording();
	record(-1);
}

void ViconRTClient::stopRecording() {
	if (m_recording) {
		m_recording = false;
		m_count = 0;
		closeFile();
	}
}

void ViconRTClient::openFile() {
	std::ostringstream os;
	unsigned int current = TimingHelper::getCurrentTimeMillis();
	os << m_prefix << "_" << current;
	m_filename = os.str();
	m_tmpfileBodies.open("tmp_vicon.bvh", std::ofstream::trunc);
	m_tmpfileBodies << std::fixed << std::setprecision(5);
	m_tmpfileMarkers.open("tmp_vicon.mkr", std::ofstream::trunc);
	m_tmpfileMarkers << std::fixed << std::setprecision(5);
	m_tmpcount = 0;
}

void ViconRTClient::closeFile() {
	if (m_tmpfileBodies.is_open()) {
		m_tmpfileBodies.close();
		m_fileBodies.open(m_filename + ".bvh", std::ofstream::trunc);
		if (m_fileBodies.is_open()) {
			m_fileBodies << std::fixed << std::setprecision(5);
			m_fileBodies << "HIERARCHY\n";
			for (unsigned int i = 0;i < m_skeletons.size();i++) {
				writeSkeleton(m_skeletons[i]);
			}
			m_fileBodies << "MOTION\n";
			m_fileBodies << "Frames: " << m_tmpcount << "\n";
			m_fileBodies << "Frame Time: " << 1.0f / 120.0f << "\n";
			std::ifstream ifile;
			ifile.open("tmp_vicon.bvh");
			if (ifile.is_open()) {
				m_fileBodies << ifile.rdbuf();
				ifile.close();
			}
			m_fileBodies.close();
		}
	}
	if (m_tmpfileMarkers.is_open()) {
		m_tmpfileMarkers.close();
		m_fileMarkers.open(m_filename + ".mkr", std::ofstream::trunc);
		if (m_fileMarkers.is_open()) {
			m_fileMarkers << std::fixed << std::setprecision(5);
			writeMarker();
			m_fileMarkers << "MOTION\n";
			m_fileMarkers << "Frames: " << m_tmpcount << "\n";
			m_fileMarkers << "Frame Time: " << 1.0f / 120.0f << "\n";
			std::ifstream ifile;
			ifile.open("tmp_vicon.mkr");
			if (ifile.is_open()) {
				m_fileMarkers << ifile.rdbuf();
				ifile.close();
			}
			m_fileMarkers.close();
		}
	}
}


void ViconRTClient::populateSkeleton(BoneData *parent) {
	std::vector<BoneData>::iterator iBoneData;
	for (iBoneData = Bones.begin();iBoneData != Bones.end();iBoneData++) {
		std::size_t found = iBoneData->channel->Name.find("Root");
		if (found != std::string::npos) continue;
		/*	if(iBoneData->visit==false&&iBoneData->body->btX==iBoneData->body->TX&&iBoneData->body->btY==iBoneData->body->TY&&iBoneData->body->btZ==iBoneData->body->TZ){
				//look like another hierarchy with a different root is present
				continue;

			}*/
		if (!iBoneData->visit) {
			Eigen::Affine3d pLocal;
			pLocal = parent->toWorld.inverse()*iBoneData->toWorld;
			bool approx = iBoneData->toLocal.isApprox(pLocal, 0.1f);

			/*if(fabs(iBone->body->btX-iBone->body->TX+parent->body->TX)<0.1f&&
				fabs(iBone->body->btY-iBone->body->TY+parent->body->TY)<0.1f&&
				fabs(iBone->body->btZ-iBone->body->TZ+parent->body->TZ)<0.1f){*/
			if (approx) {
				printf("Bone Parent %s is %s\n", iBoneData->channel->Name.c_str(), parent->channel->Name.c_str());
				parent->children.push_back(&(*iBoneData));
				iBoneData->visit = true;
			}
		}
	}
	if (parent->children.size() > 0) {
		for (std::vector<BoneData *>::iterator iChildData = parent->children.begin();iChildData != parent->children.end();iChildData++) {
			populateSkeleton(*iChildData);
		}
	}
}

void ViconRTClient::initSkeleton() {
	std::vector< BodyChannel >::iterator iBody;
	std::vector< BodyData >::iterator iBodyData;
	std::vector< BoneData>::iterator iBoneData;
	std::vector< MarkerData>::iterator iMarkerData;

	for (iBoneData = Bones.begin();iBoneData != Bones.end();iBoneData++) {
		iBoneData->visit = false;
	}

	do {
		for (iBoneData = Bones.begin();iBoneData != Bones.end();iBoneData++) {
			if (iBoneData->visit == false) {
				std::size_t found = iBoneData->channel->Name.find("Root");
				if (found != std::string::npos) break;
			}
		}
		/*for(iBoneData = Bones.begin();iBoneData != Bones.end();iBoneData++){
			if(iBoneData->visit==false&&iBoneData->body->btX==iBoneData->body->TX&&iBoneData->body->btY==iBoneData->body->TY&&iBoneData->body->btZ==iBoneData->body->TZ){
				break;
			}
		}*/
		if (iBoneData == Bones.end()) break;

		BoneData *skeleton = &(*iBoneData);
		skeleton->visit = true;
		m_skeletons.push_back(skeleton);
		printf("Skeleton Root is %s\n", skeleton->channel->Name.c_str());
		populateSkeleton(skeleton);
		defineBonesOrder(skeleton);
	} while (true);
	m_initSkeleton = true;
}

void ViconRTClient::defineBonesOrder(BoneData *node) {
	std::vector< BoneData>::const_iterator iBoneData;
	for (iBoneData = Bones.begin();iBoneData != Bones.end();iBoneData++) {
		if (iBoneData->channel->Name == node->channel->Name) {
			break;
		}
	}
	BonesOrder.push_back(iBoneData - Bones.begin());
	for (std::vector<BoneData *>::iterator iChildData = node->children.begin();iChildData != node->children.end();iChildData++) {
		defineBonesOrder(*iChildData);
	}
}

void ViconRTClient::update() {
	if (!m_stop&&!m_initSkeleton)
		initSkeleton();
	else if (!m_initSkeleton) return;

	Eigen::Affine3f matrix;

	for (std::vector< BoneData >::const_iterator iBoneData = Bones.begin();
	iBoneData != Bones.end();iBoneData++) {
		matrix = iBoneData->toLocal.cast<float>();
		//setLocalToWorldMatrix were you want
	}
	for (std::vector< MarkerData >::const_iterator iMarkerData = markerPositions.begin();
	iMarkerData != markerPositions.end(); iMarkerData++) {
		matrix = Eigen::Translation3f(iMarkerData->X, iMarkerData->Y, iMarkerData->Z);
		//setLocalToWorldMatrix were you want
	}
}

void ViconRTClient::writeMarker() {
	m_fileMarkers << "MARKERS\n";
	m_fileMarkers << "CHANNELS 4 Xposition Yposition Zposition Visibility" << "\n";
	m_fileMarkers << "{\n";
	for (std::vector< MarkerChannel>::iterator iMarker = MarkerChannels.begin();iMarker != MarkerChannels.end(); iMarker++) {
		m_fileMarkers << "\tMARKER " << iMarker->Name << "\n";
	}
	m_fileMarkers << "}\n";
}

void ViconRTClient::writeSkeleton(BoneData *node, BoneData *parent, unsigned int level) {
	if (node == NULL) return;

	if (parent == NULL) {
		m_fileBodies << "ROOT " << node->channel->Name << std::endl;
		level = 0;
	}
	else {
		for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
		m_fileBodies << "JOINT " << node->channel->Name << "\n";
	}
	for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
	m_fileBodies << "{\n";
	for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
	if (parent != NULL) {
		Eigen::Vector3d pos = Eigen::Vector3d(node->body->TX, node->body->TY, node->body->TZ);
		pos = parent->toWorld.inverse()*pos;
		//m_file<<"\tOFFSET\t"<<node->body->TX-parent->body->TX<<"\t"<<node->body->TY-parent->body->TY<<"\t"<<node->body->TZ-parent->body->TZ<<"\n";
		m_fileBodies << "\tOFFSET\t" << pos.x() << "\t" << pos.y() << "\t" << pos.z() << "\n";
	}
	else m_fileBodies << "\tOFFSET\t0.00\t0.00\t0.00\n";
	for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
	m_fileBodies << "\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n";
	if (node->children.size() > 0)
		for (std::vector<BoneData *>::iterator iChildData = node->children.begin();iChildData != node->children.end();iChildData++) writeSkeleton(*iChildData, node, level + 1);
	else {
		for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
		m_fileBodies << "\tEnd Site \n";
		for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
		m_fileBodies << "\t{\n";
		for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
		m_fileBodies << "\t\tOFFSET\t0.00\t20.00\t0.00\n";
		for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
		m_fileBodies << "\t}\n";
	}
	for (unsigned int i = 0;i < level;i++) m_fileBodies << "\t";
	m_fileBodies << "}\n";
}

void ViconRTClient::recordFileData() {
	if (m_recording) {
		if (m_count > 0) {
			m_count--;
			if (m_count == 0) m_recording = false;
		}
		if (m_tmpfileBodies.is_open()) {
			for (std::vector<unsigned int>::const_iterator it = BonesOrder.begin();it != BonesOrder.end();it++) {
				std::vector< BodyData >::iterator iBodyData = bodyPositions.begin() + (*it);
				m_tmpfileBodies << iBodyData->btX << " " << iBodyData->btY << " " << iBodyData->btZ << " " << iBodyData->beulerX*180.0 / M_PI << " " << iBodyData->beulerY*180.0 / M_PI << " " << iBodyData->beulerZ*180.0 / M_PI << " ";
				//m_file<<iBodyData->TX<<" "<<iBodyData->TY<<" "<<iBodyData->TZ<<" "<<iBodyData->EulerX<<" "<<iBodyData->EulerY<<" "<<iBodyData->EulerZ<<" ";
			}
			m_tmpfileBodies << "\n";
		}
		if (m_tmpfileMarkers.is_open()) {
			for (std::vector<MarkerData>::const_iterator iMarkerData = markerPositions.begin();iMarkerData != markerPositions.end();iMarkerData++) {
				m_tmpfileMarkers << iMarkerData->X << " " << iMarkerData->Y << " " << iMarkerData->Z << " " << (iMarkerData->Visible ? 1.0f : 0.0f) << " ";
			}
			m_tmpfileMarkers << "\n";
		}
		m_tmpcount++;
		if (m_count == 0) closeFile();
	}
}

void ViconRTClient::printData() const {
	std::cout << "All Bodies:\n";
	//for(std::vector<unsigned int>::const_iterator it=BonesOrder.begin();it!=BonesOrder.end();it++){
		//std::vector< BodyData >::iterator iBodyData=bodyPositions.begin()+(*it);
	for (std::vector< BodyData >::const_iterator iBodyData = bodyPositions.begin();iBodyData != bodyPositions.end();iBodyData++) {
		std::cout << iBodyData->btX << " " << iBodyData->btY << " " << iBodyData->btZ << " " << iBodyData->beulerX*180.0 / M_PI << " " << iBodyData->beulerY*180.0 / M_PI << " " << iBodyData->beulerZ*180.0 / M_PI << " ";
	}
	std::cout << "\n";
	std::cout << "All Markers:\n";
	for (std::vector<MarkerData>::const_iterator iMarkerData = markerPositions.begin();iMarkerData != markerPositions.end();iMarkerData++) {
		std::cout << iMarkerData->X << " " << iMarkerData->Y << " " << iMarkerData->Z << " " << (iMarkerData->Visible ? 1.0f : 0.0f) << " ";
	}
	std::cout << "\n";
}

#pragma warning(pop)