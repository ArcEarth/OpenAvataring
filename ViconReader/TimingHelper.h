#ifndef TIMING_H
#define TIMING_H

#if defined(__CELLOS_LV2__) || defined(_XBOX) || defined(LINUX)
	unsigned long timeGetTime();
#elif defined(WIN32) || defined(_WIN64)

#ifndef NOMINMAX
#	define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#	define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX
#endif

#if defined(__CELLOS_LV2__)

#include <sys/sys_time.h>
#include <sys/time_util.h>

typedef union _LARGE_INTEGER {
    uint64_t QuadPart;
} LARGE_INTEGER;

inline void QueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount){
	SYS_TIMEBASE_GET(lpPerformanceCount->QuadPart);
}

inline void QueryPerformanceFrequency(LARGE_INTEGER *lpFrequency){
	lpFrequency->QuadPart = sys_time_get_timebase_frequency();
}

#endif //defined(__CELLOS_LV2__)

#if defined(LINUX)

#include <stdint.h>
#include <time.h>

typedef union _LARGE_INTEGER {
	uint64_t QuadPart;
} LARGE_INTEGER;

inline void QueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount){
	lpPerformanceCount->QuadPart = clock();
}

inline void QueryPerformanceFrequency(LARGE_INTEGER *lpFrequency){
	lpFrequency->QuadPart = CLOCKS_PER_SEC;
}

#endif // defined(LINUX)

class TimingHelper {
public:
	static float getCurrentTime();
	static unsigned int getCurrentTimeMillis();
	static float getElapsedTime();
	static void resetCElapsedTimeCounter();
	static float getCElapsedTime();
	static float getCElapsedTimeMillis();
	static float GetSystemTimeSeconds();
};

#endif
