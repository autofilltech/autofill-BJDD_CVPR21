#include <torch/extension.h>
#include <cuda_runtime.h>

#include "operators.h"

template <typename T>
__global__ void f_unpack12(T* dst, uint8_t* src, uint32_t width, uint32_t height)
{
	assert(0 == (width & 0x1F));

	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	typedef struct { T x,y; } T2;

	for (auto row = offset.y; row < height; row += stride.y)
	{
		for (auto col = offset.x; col < width/2; col += stride.x)
		{
			auto s1 = ((uchar3*)src)[row * width/2 + col];
			T2 s2 = {
					(((s1.x << 4) | (s1.y >> 4)) & 0xFFF) / (T) 0xFFF,
					(((s1.y << 8) | (s1.z     )) & 0xFFF) / (T) 0xFFF,
			};
			((T2*)dst)[row * width/2 + col] = s2;
		}
	}
}

template <typename T>
__global__ void f_pack12(uint8_t* dst, T* src, uint32_t width, uint32_t height)
{
	assert(0 == (width & 0x1F));

	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;
	
	typedef struct { T x,y; } T2;

	for (auto row = offset.y; row < height; row += stride.y)
	{
		for (auto col = offset.x; col < width/2; col += stride.x)
		{
			auto s1 = ((T2*)src)[row * width/2 + col];
			auto s2 = make_ushort2(
					__saturatef(s1.x) * 0xFFF,
					__saturatef(s1.y) * 0xFFF);
			auto s3 = make_uchar3(
					(s2.x >> 4) & 0xFF,
					((s2.x << 4) | (s2.y >> 8)) & 0xFF,
					s2.y & 0xFF);
			((uchar3*)dst)[row * width/2 + col] = s3;
		}
	}
}

template <typename T>
void pack12(uint8_t* dst, T* src, uint32_t width, uint32_t height, cudaStream_t stream)
{
	f_pack12 <<< {32,32}, {16,16}, 0, stream >>> (dst, src, width, height);
}

template <typename T>
void unpack12(T* dst, uint8_t* src, uint32_t width, uint32_t height, cudaStream_t stream)
{
	f_unpack12 <<< {32,32}, {16,16}, 0, stream >>> (dst, src, width, height);
}

template void pack12<float>(uint8_t* dst, float* src, uint32_t width, uint32_t height, cudaStream_t stream);
template void pack12<double>(uint8_t* dst, double* src, uint32_t width, uint32_t height, cudaStream_t stream);
template void unpack12<float>(float* dst, uint8_t* src, uint32_t width, uint32_t height, cudaStream_t stream);
template void unpack12<double>(double* dst, uint8_t* src, uint32_t width, uint32_t height, cudaStream_t stream);

