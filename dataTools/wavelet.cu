#include "wavelet.h"
#include <assert.h>

template <typename T, size_t Support>
__global__ static void f_wavelet_horizontal_pass(T* data, uint32_t stride, uint32_t width, uint32_t height, T threshold = 0)
{
	assert(1 == blockDim.y);
	assert(1 == gridDim.x);
	assert(gridDim.y == height);

	Daubechies<Support> w;

	auto row = blockIdx.y;
	extern __shared__ uint8_t shm[];
	T* mem = (T*) shm; 

	for (auto col = threadIdx.x; col < width; col += blockDim.x) mem[col] = data[row * stride + col];
	__syncthreads();
	for (auto col = threadIdx.x; col < width/2; col += blockDim.x)
	{
		T p[Support];
		#pragma unroll
		for (auto i = 0U; i < Support; i++) p[i] = mem[(col*2 + i) % width];
		auto L = w.scaling(p)/2;
		auto H = w.wavelet(p)/2;
		data[row * stride + col] = L;
		data[row * stride + col + width/2] = fabs(H) > threshold ? H + 0.5 : 0.5;
	}
}

// TODO naive impl. not coalesced, optimize
template <typename T, size_t Support>
__global__ static void f_wavelet_vertical_pass(T* data, uint32_t stride, uint32_t width, uint32_t height, T threshold = 0)
{
	assert(1 == blockDim.x);
	assert(1 == gridDim.y);
	assert(gridDim.x == width);

	Daubechies<Support> w;

	auto col = blockIdx.x;
	extern __shared__ uint8_t shm[];
	T* mem = (T*) shm; 

	for (auto row = threadIdx.y; row < height; row += blockDim.y) mem[row] = data[row * stride + col];
	__syncthreads();
	for (auto row = threadIdx.y; row < height/2; row += blockDim.y)
	{
		T p[Support];
		#pragma unroll
		for (auto i = 0U; i < Support; i++) p[i] = mem[(row*2 + i) % height];
		auto L = w.scaling(p)/2;
		auto H = w.wavelet(p)/2;
		data[row * stride + col] = L;
		data[(row + height/2) * stride + col] = fabs(H) > threshold ? H+0.5 : 0.5;
	}
}


template <typename T, size_t Support>
__global__ static void f_wavelet_horizontal_inv_pass(T* data, uint32_t stride, uint32_t width, uint32_t height)
{
	assert(1 == blockDim.y);
	assert(1 == gridDim.x);
	assert(gridDim.y == height);

	Daubechies<Support> w;

	auto row = blockIdx.y;
	extern __shared__ uint8_t shm[];
	T* mem = (T*) shm; 

	for (auto col = threadIdx.x; col < width; col += blockDim.x) mem[col] = data[row * stride + col];
	__syncthreads();
	for (auto col = threadIdx.x; col < width/2; col += blockDim.x)
	{
		T p[Support];
		#pragma unroll
		for (auto i = 0U; i < Support; i++) 
		{
			auto offset = Support/2 - 1;
			p[i] = mem[((col + (i + width)/2 - offset) % (width/2)) + ((i&1) ? width/2 : 0)];
			if (i&1) p[i] -= 0.5;
		}
		data[row * stride + 2 * col + 0] = w.scalingInv(p);
		data[row * stride + 2 * col + 1] = w.waveletInv(p);
	}
}

// TODO naive impl. not coalesced, optimize
template <typename T, size_t Support>
__global__ static void f_wavelet_vertical_inv_pass(T* data, uint32_t stride, uint32_t width, uint32_t height)
{
	assert(1 == blockDim.x);
	assert(1 == gridDim.y);
	assert(gridDim.x == width);

	Daubechies<Support> w;
	
	auto col = blockIdx.x;
	extern __shared__ uint8_t shm[];
	T* mem = (T*) shm; 

	for (auto row = threadIdx.y; row < height; row += blockDim.y) mem[row] = data[row * stride + col];
	__syncthreads();
	for (auto row = threadIdx.y; row < height/2; row += blockDim.y)
	{
		T p[Support];
		#pragma unroll
		for (auto i = 0U; i < Support; i++) 
		{
			auto offset = Support/2 - 1;
			p[i] = mem[((row + (i + height)/2 - offset) % (height/2)) + ((i&1) ? height/2 : 0)];
			if (i&1) p[i] -= 0.5;
		}
		data[(2 * row + 0) * stride + col] = w.scalingInv(p);
		data[(2 * row + 1) * stride + col] = w.waveletInv(p);
	}
}

template <size_t Support>
template <typename T>
void Daubechies<Support>::transform(T* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream)
{
	assert(width * sizeof(T) <= 1<<16);
	assert(height * sizeof(T) <= 1<<16);
	
	f_wavelet_horizontal_pass <T,Support> <<< {1, height}, {1024,1}, width * sizeof(T), stream >>> ( 
			data,
			stride,
			width,
			height);

	f_wavelet_vertical_pass <T,Support> <<< {width,1}, {1,1024}, height * sizeof(T), stream >>> (
			data,
			stride,
			width,
			height);
}

template <size_t Support>
template <typename T>
void Daubechies<Support>::transformWithThreshold(T* data, uint32_t stride, uint32_t width, uint32_t height, T threshold, cudaStream_t stream)
{
	assert(width * sizeof(T) <= 1<<16);
	assert(height * sizeof(T) <= 1<<16);
	
	f_wavelet_horizontal_pass <T,Support> <<< {1, height}, {1024,1}, width * sizeof(T), stream >>> ( 
			data,
			stride,
			width,
			height,
			threshold);
	f_wavelet_vertical_pass <T,Support> <<< {width,1}, {1,1024}, height * sizeof(T), stream >>> (
			data,
			stride,
			width,
			height,
			threshold);
}

template <size_t Support>
template <typename T>
void Daubechies<Support>::transformInv(T* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream)
{
	assert(width * sizeof(T) <= 1<<16);
	assert(height * sizeof(T) <= 1<<16);

	f_wavelet_horizontal_inv_pass <T,Support> <<< {1, height}, {1024,1}, width * sizeof(T), stream >>> ( 
			data,
			stride,
			width,
			height);
	f_wavelet_vertical_inv_pass <T,Support> <<< {width,1}, {1,1024}, height * sizeof(T), stream >>> (
			data,
			stride,
			width,
			height);
}

template void Daubechies<2>::transform<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<4>::transform<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<6>::transform<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<8>::transform<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<2>::transform<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<4>::transform<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<6>::transform<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<8>::transform<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);

template void Daubechies<2>::transformInv<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<4>::transformInv<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<6>::transformInv<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<8>::transformInv<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<2>::transformInv<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<4>::transformInv<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<6>::transformInv<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);
template void Daubechies<8>::transformInv<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream);

template void Daubechies<2>::transformWithThreshold<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, float threshold, cudaStream_t stream);
template void Daubechies<4>::transformWithThreshold<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, float threshold, cudaStream_t stream);
template void Daubechies<6>::transformWithThreshold<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, float threshold, cudaStream_t stream);
template void Daubechies<8>::transformWithThreshold<float>(float* data, uint32_t stride, uint32_t width, uint32_t height, float threshold, cudaStream_t stream);
template void Daubechies<2>::transformWithThreshold<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, double threshold, cudaStream_t stream);
template void Daubechies<4>::transformWithThreshold<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, double threshold, cudaStream_t stream);
template void Daubechies<6>::transformWithThreshold<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, double threshold, cudaStream_t stream);
template void Daubechies<8>::transformWithThreshold<double>(double* data, uint32_t stride, uint32_t width, uint32_t height, double threshold, cudaStream_t stream);

template <> __device__ 
Daubechies<2>::Daubechies()
{
	v[0] = 1;
	v[1] = 1;
}

template <> __device__
Daubechies<4>::Daubechies()
{
	v[0] = 0.6830127;
	v[1] = 1.1830127;
	v[2] = 0.3169873;
	v[3] =-0.1830127;
}

template <> __device__
Daubechies<6>::Daubechies()
{
	v[0] = 0.47046721;
	v[1] = 1.14111692;
	v[2] = 0.65036500;
	v[3] =-0.19093442;
	v[4] =-0.12083221;
	v[5] = 0.04981750;
}

template <> __device__
Daubechies<8>::Daubechies()
{
	v[0] = 0.32580343;
	v[1] = 1.01094572;
	v[2] = 0.89220014;
	v[3] =-0.03957503;
	v[4] =-0.26450717;
	v[5] = 0.04361630;
	v[6] = 0.04650360;
	v[7] =-0.01498699;
}

template struct Daubechies<2>;
template struct Daubechies<4>;
template struct Daubechies<6>;
template struct Daubechies<8>;
