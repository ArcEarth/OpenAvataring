#include <stdio.h>
#ifdef WIN32 
#	define NOMINMAX
#	include <windows.h>
#endif

#if defined(_XBOX)
#	include <xtl.h>
#endif

#include "TimingHelper.h"

#if defined(__CELLOS_LV2__)

#include <sys/sys_time.h>
#include <sys/time_util.h>


unsigned long timeGetTime()
{
	static uint64_t ulScale=0;
	uint64_t ulTime;

	if (ulScale==0) {
		ulScale = sys_time_get_timebase_frequency() / 1000;
	}

#ifdef __SNC__
	ulTime=__builtin_mftb();
#else
	asm __volatile__ ("mftb %0" : "=r" (ulTime) : : "memory");
#endif

	return ulTime/ulScale;
}

#elif defined(LINUX)

#include <sys/time.h>

BEGIN_NAMESPACE_VIMMI3D

unsigned long timeGetTime()
{
        timeval tim;
	gettimeofday(&tim, NULL);
	unsigned long ms = (tim.tv_sec*1000u)+(long)(tim.tv_usec/1000.0);
	return ms;
}

#elif defined(_XBOX)

BEGIN_NAMESPACE_VIMMI3D

unsigned long timeGetTime()
{
	LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    unsigned long long ticksPerMillisecond = freq.QuadPart/1000;

	LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (unsigned long)(counter.QuadPart/ticksPerMillisecond);
}
#endif

#if !defined(LINUX)

static LARGE_INTEGER freq;
static bool freqinit = false;
static LARGE_INTEGER fstartTime; 
static LARGE_INTEGER fpreviousTime;

unsigned int TimingHelper::getCurrentTimeMillis()
{
	return (unsigned int)(getCurrentTime()*1000.0f);
}

float TimingHelper::getCurrentTime()
{
	if(!freqinit){
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&fstartTime);
		fpreviousTime=fstartTime;
		freqinit=true;
	}
	LARGE_INTEGER currentTime;
	QueryPerformanceCounter(&currentTime);
	return (float)((currentTime.QuadPart)/long double(freq.QuadPart));
}

float TimingHelper::getElapsedTime()
{
	if(!freqinit){
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&fstartTime);
		fpreviousTime=fstartTime;
		freqinit=true;
	}
	LARGE_INTEGER currentTime;
	QueryPerformanceCounter(&currentTime);
	unsigned long long elapsedTime = currentTime.QuadPart - fpreviousTime.QuadPart;
	fpreviousTime = currentTime;
	return (float)(elapsedTime)/(freq.QuadPart);
}



void TimingHelper::resetCElapsedTimeCounter(){
	QueryPerformanceCounter(&fstartTime);
}

float TimingHelper::getCElapsedTime(){
	if(!freqinit){
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&fstartTime);
		fpreviousTime=fstartTime;
		freqinit=true;
	}
	LARGE_INTEGER currentTime;
	QueryPerformanceCounter(&currentTime);
	unsigned long long elapsedTime = currentTime.QuadPart - fstartTime.QuadPart;
	return (float)(elapsedTime)/(freq.QuadPart);
}

float TimingHelper::getCElapsedTimeMillis(){
	return getCElapsedTime()*1000.0f;
}

float TimingHelper::GetSystemTimeSeconds(){ 
	SYSTEMTIME st;
	GetSystemTime(&st);
	float totalSec=st.wHour*3600.0f+st.wMinute*60.0f+ st.wSecond + st.wMilliseconds*0.001f;
	return totalSec;
}

#else

float TimingHelper::getElapsedTime()
{
	static timeval previousTime;
	static bool init = false;
	if(!init){
 	        gettimeofday(&previousTime, NULL);
		init=true;
	}
	timeval currentTime;
	gettimeofday(&currentTime, NULL);
	double elapsedTime = (currentTime.tv_sec+currentTime.tv_usec/1000000.0) - (previousTime.tv_sec+previousTime.tv_usec/1000000.0);
	previousTime = currentTime;
	return (float)elapsedTime;
}

#endif
