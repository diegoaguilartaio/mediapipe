#pragma once
 #include <cstdint>
constexpr int STR_PARAM_SIZE = 50;
#if defined __clang__
#define PACK_DECORATOR __attribute__((packed))
#else
#define PACK_DECORATOR
#endif


typedef struct  PACK_DECORATOR
{
	const char *Data;
	int Width;
	int Height;
	int64_t FrameId;
}FrameInfo;