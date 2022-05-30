#include "shuffle.h"
template <typename T>
__global__ void f_unshuffle4(
		T* r, T* gr, T* gb, T* b, 
		uint32_t dst_stride, 
		T* src, 
		uint32_t src_stride, 
		uint32_t width, uint32_t height)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto row=offset.y; row < height/2; row += stride.y)
	{
		auto s0 = src + (row*2 + 0) * src_stride;
		auto s1 = src + (row*2 + 1) * src_stride;
		auto r0  = r  + (row) * dst_stride;
		auto gr0 = gr + (row) * dst_stride;
		auto gb0 = gb + (row) * dst_stride;
		auto b0  = b  + (row) * dst_stride;
		
		for (auto col=offset.x; col < width/2; col += stride.x)
		{
			typedef struct { T x,y; } T2;
			T2 v0 = ((T2*) s0)[col];
			T2 v1 = ((T2*) s1)[col];

			r0 [col] = v0.x;
			gr0[col] = v0.y;
			gb0[col] = v1.x;
			b0 [col] = v1.y;
		}
	}

}

template <typename T>
__global__ void f_unshuffle(T* dst, uint32_t dst_stride, T* src, uint32_t src_stride, uint32_t width, uint32_t height)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto row=offset.y; row < height/2; row += stride.y)
	{
		auto s0 = src + (row*2 + 0) * src_stride;
		auto s1 = src + (row*2 + 1) * src_stride;
		auto d0 = dst + (row) * dst_stride;
		auto d1 = dst + (row + height/2) * dst_stride;
		
		for (auto col=offset.x; col < width/2; col += stride.x)
		{
			typedef struct { T x,y; } T2;
			T2 v0 = ((T2*) s0)[col];
			T2 v1 = ((T2*) s1)[col];

			d0[col] = v0.x;
			d0[col + width/2] = v0.y;
			d1[col] = v1.x;
			d1[col + width/2] = v1.y;
		}
	}
}

template <typename T>
__global__ void f_shuffle4(
		T* dst, 
		uint32_t dst_stride, 
		T* r, T* gr, T* gb, T* b, 
		uint32_t src_stride, 
		uint32_t width, uint32_t height)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto row=offset.y; row < height; row += stride.y)
	{
		auto r0 = r + (row) * src_stride; 
		auto gr0 = gr + (row) * src_stride;
		auto gb0 = gb + (row) * src_stride;
		auto b0 = b + (row) * src_stride;


		auto d0 = dst + row * 2 * dst_stride;
		auto d1 = dst + (row + 1) * 2 * dst_stride;
		
		for (auto col=offset.x; col < width; col += stride.x)
		{
			typedef struct { T x,y; } T2;
			T2 v0 = {
				.x = r0[col],
				.y = gr0[col]
			};
			T2 v1 = {
				.x = gb0[col],
				.y = b0[col]
			};
		
			((T2*)d0)[col] = v0;
			((T2*)d1)[col] = v1;
		}
	}
}
template <typename T>
__global__ void f_shuffle(T* dst, uint32_t dst_stride, T* src, uint32_t src_stride, uint32_t width, uint32_t height)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto row=offset.y; row < height/2; row += stride.y)
	{
		auto d0 = src + (row*2 + 0) * src_stride; 
		auto d1 = src + (row*2 + 1) * src_stride;

		auto s0 = dst + (row) * dst_stride;
		auto s1 = dst + (row + height/2) * dst_stride;
		
		for (auto col=offset.x; col < width/2; col += stride.x)
		{
			typedef struct { T x,y; } T2;
			T2 v0 = {
				.x = s0[col],
				.y = s0[col + width/2]
			};
			T2 v1 = {
				s1[col],
				s1[col + width/2]
			};
		
			((T2*)d0)[col] = v0;
			((T2*)d1)[col] = v1;
		}
	}
}

torch::Tensor shuffle4(torch::Tensor src)
{
	assert(src.size(0) == 4);
	assert(src.size(1));
	assert(src.size(2));
	
	auto height = src.size(1);
	auto width = src.size(2);
	
	assert(src.stride(1) >= width);
	assert(src.stride(2) == 1);
	assert(src.dtype() == torch::kFloat32);

	float* dst;
	int rc = cudaMalloc(&dst, width * height * 4 * sizeof(float));
	assert(cudaSuccess == rc);

	auto rstride = width * 2;

	f_shuffle4 <<< {32,32}, {16,16}, 0, 0 >>> (
			dst, rstride,
			src[0].data_ptr<float>(), 
			src[1].data_ptr<float>(), 
			src[2].data_ptr<float>(), 
			src[3].data_ptr<float>(), 
			width,
			width, height);

	auto options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.device(torch::kCUDA, CUDA_DEVICE_IDX);
	return torch::from_blob(dst, { 1, height * 2, width * 2 }, { width * height * 4, rstride, 1 }, cudaFree, options);
}

torch::Tensor shuffle(torch::Tensor src)
{
	assert(src.size(0) == 1);
	assert(src.size(1));
	assert(src.size(2));
	
	auto height = src.size(1);
	auto width = src.size(2);
	
	assert(src.stride(1) >= width);
	assert(src.stride(2) == 1);
	assert(src.dtype() == torch::kFloat32);

	float* dst;
	int rc = cudaMalloc(&dst, width * height * sizeof(float));
	assert(cudaSuccess == rc);

	f_shuffle <<< {32,32}, {16,16}, 0, 0 >>> (
			dst, width,
			src.data_ptr<float>(), src.stride(1),
			width, height);

	auto options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.device(torch::kCUDA, CUDA_DEVICE_IDX);
	return torch::from_blob(dst, { 1, height, width }, { width * height, width, 1 }	, cudaFree, options);
}

torch::Tensor unshuffle4(torch::Tensor src)
{
	assert(src.size(0) == 1);
	assert(src.size(1));
	assert(src.size(2));
	
	auto height = src.size(1);
	auto width = src.size(2);
	
	assert(src.stride(0) == width * height);
	assert(src.stride(1) == width);
	assert(src.stride(2) == 1);
	assert(src.dtype() == torch::kFloat32);

	float* dst;
	int rc = cudaMalloc(&dst, width * height * sizeof(float));
	
	auto cstride = (width * height) / 4;
	auto rstride = (width) / 2;

	f_unshuffle4 <<< {32,32}, {16,16}, 0, 0 >>> (
			dst, dst + cstride, dst + cstride * 2, dst + cstride * 3, 
			rstride,
			src.data_ptr<float>(), 
			src.stride(1),
			width, height);
	
	auto options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.device(torch::kCUDA, CUDA_DEVICE_IDX);
	return torch::from_blob(dst, { 4, height / 2, width / 2 }, { cstride, rstride, 1 }, cudaFree, options);
}

torch::Tensor unshuffle(torch::Tensor src)
{
	assert(src.size(0) == 1);
	assert(src.size(1));
	assert(src.size(2));
	
	auto height = src.size(1);
	auto width = src.size(2);
	
	assert(src.stride(0) == width * height);
	assert(src.stride(1) == width);
	assert(src.stride(2) == 1);
	assert(src.dtype() == torch::kFloat32);
	
	float* dst;
	int rc = cudaMalloc(&dst, width * height * sizeof(float));
	assert(cudaSuccess == rc);
	
	f_unshuffle <<< {32,32}, {16,16}, 0, 0 >>> (
			dst, width,
			src.data_ptr<float>(), src.stride(1),
			width, height);

	auto options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.device(torch::kCUDA, CUDA_DEVICE_IDX);
	return torch::from_blob(dst, { 1, height, width }, { width * height, width, 1 }	, cudaFree, options);
}
