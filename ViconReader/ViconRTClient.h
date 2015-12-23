//-----------------------------------------------------------------------------
//	Vicon RT Client
//-----------------------------------------------------------------------------
#pragma once

#include <string>
#include <iosfwd>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>	
#include <functional>
#include <limits>
#include <cmath>
#include <memory>
#include <thread>

// For UINT_PTR
#include <basetsd.h>

#include "ClientCodes.h"

const int VERBOSE = 1;
const int BRIEF = 0;
#define PR(x) cout << #x"=" << x << endl;
//#define VC_VERBOSE

class ViconRTClient: public std::enable_shared_from_this<ViconRTClient> {
public:
	typedef std::shared_ptr<ViconRTClient> Ref;
	ViconRTClient::Ref getPtr();
	static const ViconRTClient::Ref Empty;
	static ViconRTClient::Ref create(const std::string &server_ip="localhost",const std::string &prefix="test");

	ViconRTClient(const std::string &server_ip="localhost",const std::string &prefix="test");
	~ViconRTClient(void);

	void init(const std::string &server_ip,const std::string &prefix="test");
	bool start();
	void stop();

	void getChannelInfo(void);
	double getFrame(void); 
	
	void showMarkers(int) const;
	void showBodies(int) const;
	const MarkerData &getMarkerData(const std::string &name) const;
	BodyData getBodyData(std::string) const;
	int getBodyChannelID(const std::string &name) const;
	const BodyData &getBodyDataByID(int i) const;

	void record(int numframes);
	void startRecording();
	void stopRecording();

	void update();
	void printData() const;

	static const MarkerData EmptyMarkerData;

	void setPrefix(const std::string &prefix);
private:
	void openFile();
	void closeFile();
	void recordFileData();

	int initialize(const std::string &server_ip);
	int close(void);
	virtual void run();
	static void *runStatic(void *val);

	std::vector<std::string > info;
	std::vector<MarkerChannel>	MarkerChannels;
	std::vector<MarkerData> markerPositions;
	
	std::vector<BodyData> bodyPositions;
	std::vector<BodyChannel>		BodyChannels;
	void initSkeleton();
	void populateSkeleton(BoneData *parent);
	void writeSkeleton(BoneData *node=NULL,BoneData *parent=NULL,unsigned int level=0);
	void writeMarker();
	void defineBonesOrder(BoneData *node);

	int	FrameChannel;
	int TimeVChannel;
	int TimeRChannel;
	int TimeHChannel;
	int TimeMChannel;
	int TimeSChannel;
	int TimeMSChannel;
	int TimeFChannel;
	int TimeOFFChannel;

	double m_framerate;
	double m_timecodeValidity;
	double m_timecodeRate;
	double m_timecodeHours;
	double m_timecodeMinutes;
	double m_timecodeSeconds;
	double m_timecodeMSeconds;
	double m_timecodeFrame;
	double m_timecodeOffset;

	//static const int bufferSize = 2040;
	char buff[2040];
	char * pBuff;
	UINT_PTR SocketHandle;
	long int size;

	//pthread_barrier_t m_barrier;
	//pthread_t m_thread;
	std::thread m_thread;


	std::string m_server_ip;
	bool m_stop;

	std::string m_prefix;
	std::string m_filename;
	std::ofstream m_tmpfileBodies;
	std::ofstream m_fileBodies;
	std::ofstream m_tmpfileMarkers;
	std::ofstream m_fileMarkers;
	int m_count;
	int m_tmpcount;
	bool m_recording;
	bool m_initSkeleton;

	std::vector<BoneData> Bones;
	std::vector<unsigned int> BonesOrder;
	std::vector<BoneData *> m_skeletons;
};