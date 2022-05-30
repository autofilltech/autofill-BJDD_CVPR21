#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
void pack12(uint8_t* dst, T* src, uint32_t width, uint32_t height, cudaStream_t stream = 0);

template <typename T>
void unpack12(T* dst, uint8_t* src, uint32_t width, uint32_t height, cudaStream_t stream = 0);

inline uint32_t packedBufferSize(uint32_t width, uint32_t height)
{
	return (height * width * 3) / 2;
}

template <typename T>
inline int32_t unpackedBufferSize(uint32_t width, uint32_t height)
{
	return (height * width) * sizeof(T);
}
