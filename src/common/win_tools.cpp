#ifdef _WIN32

#include "../../include/win_tools.h"
#include <windows.h>

void setConsoleSize(int width, int height, int bufferHeight)
{
	_COORD coord;
	coord.X = width;
	coord.Y = bufferHeight; // height;

	_SMALL_RECT Rect;
	Rect.Top = 0;
	Rect.Left = 0;
	Rect.Bottom = height - 1;
	Rect.Right = width - 1;

	HANDLE Handle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleScreenBufferSize(Handle, coord);
	SetConsoleWindowInfo(Handle, TRUE, &Rect);
}

ctrlCFnPtr_t s_ctrlcFn = nullptr;

BOOL WINAPI consoleHandler(DWORD signal)
{
	if (signal == CTRL_C_EVENT) {
		if (s_ctrlcFn) {
			s_ctrlcFn();
		}
	}
	return TRUE;
}

bool setCtrlCHandler(ctrlCFnPtr_t pFn) {	
	if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
		return false;
	}
	s_ctrlcFn = pFn;
	return true;
}

int gettimeofday(struct timeval* p, void* tz) {
    ULARGE_INTEGER ul; // As specified on MSDN.
    FILETIME ft;

    // Returns a 64-bit value representing the number of
    // 100-nanosecond intervals since January 1, 1601 (UTC).
    GetSystemTimeAsFileTime(&ft);

    // Fill ULARGE_INTEGER low and high parts.
    ul.LowPart = ft.dwLowDateTime;
    ul.HighPart = ft.dwHighDateTime;
    // Convert to microseconds.
    ul.QuadPart /= 10ULL;
    // Remove Windows to UNIX Epoch delta.
    ul.QuadPart -= 11644473600000000ULL;
    // Modulo to retrieve the microseconds.
    p->tv_usec = (long)(ul.QuadPart % 1000000LL);
    // Divide to retrieve the seconds.
    p->tv_sec = (long)(ul.QuadPart / 1000000LL);

    return 0;
}

#endif
