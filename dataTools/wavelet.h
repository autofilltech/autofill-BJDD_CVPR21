#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

template <size_t Support>
struct Daubechies
{
	__device__
	Daubechies() { assert(false); }

	float v[Support];

	template <typename T>
	__device__
	float scaling(const T p[Support])
	{
		float r = 0;
		#pragma unroll
		for (auto i = 0U; i < Support; i++) 
		{
			r += p[i] * v[i];
		}
		return r;
	}

	template <typename T>
	__device__
	float wavelet(const T* p)
	{
		float r = 0;
		#pragma unroll
		for (auto i = 0U; i < Support; i++) r += ((i&1) ? -p[i] : p[i]) * v[Support - 1 - i];
		return r;
	}
	template <typename T>
	__device__
	float scalingInv(const T* p)
	{
		float r = 0;
		#pragma unroll
		for (auto i = 0U; i < Support; i++) r += ((i&1) ? v[i] : v[Support-2-i]) * p[i];
		return r;	
	}
	template <typename T>
	__device__
	float waveletInv(const T* p)
	{
		float r = 0;
		#pragma unroll
		for (auto i = 0U; i < Support; i++) r += ((i&1) ? -v[i-1] : v[Support-1-i]) * p[i];
		return r;	

	}

	template <typename T>
	__host__
	static void transform(T* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream = 0);
	
	template <typename T>
	__host__
	static void transformWithThreshold(T* data, uint32_t stride, uint32_t width, uint32_t height, T threshold, cudaStream_t stream = 0);
	
	template <typename T>
	__host__
	static void transformInv(T* data, uint32_t stride, uint32_t width, uint32_t height, cudaStream_t stream = 0);
};

