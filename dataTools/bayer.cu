#include "bayer.h"

template <typename T>
__global__ void f_bayer(T* dst, T* r, T* g, T* b, uint32_t width, uint32_t height)
{
	auto stride = gridDim * blockDim;
	auto offset = blockDim * blockIdx + threadIdx;

	for (auto row = offset.y; row < height/2; row+=stride.y)
	{
		for (auto col = offset.x; col < width/2; col += stride.x)
		{
			auto vr  = r[row * 2 * width + col * 2];
			auto vgr = g[row * 2 * width + col * 2 + 1];
			auto vgb = g[(row * 2 + 1) * width + col * 2];
			auto vb  = b[(row * 2 + 1) * width + col * 2 + 1];

			dst[row * 2 * width + col * 2] = vr;
			dst[row * 2 * width + col * 2 + 1] = vgr;
			dst[(row * 2 + 1) * width + col * 2] = vgb;
			dst[(row * 2 + 1) * width + col * 2 + 1] = vb;
		}
	}
}

torch::Tensor bayer(torch::Tensor src)
{
	assert(src.size(0) == 3);
	assert(src.size(1));
	assert(src.size(2));
	assert(src.dtype() == torch::kFloat32);

	auto height = src.size(1);
	auto width = src.size(2);
	float* dst;
	int rc = cudaMalloc(&dst, width * height * sizeof(float));
	assert(cudaSuccess == rc);

	f_bayer <<< {32,32}, {16,16}, 0, 0 >>>(
			dst, 
			src[0].data<float>(), 
			src[1].data<float>(), 
			src[2].data<float>(), 
			width, height);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
	return torch::from_blob(dst, { 1, height, width }, {width * height, width, 1 }, cudaFree, options);
}
