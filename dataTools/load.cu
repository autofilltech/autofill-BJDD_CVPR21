#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/lz4.hpp>
#include <iostream>
#include <string>

#include "operators.h"
#include "wavelet.h"
#include "pack12.h"
#include "shuffle.h"
#include "bayer.h"


void packM12(std::string path, torch::Tensor t)
{
	assert(t.size(0) == 1);
	assert(t.size(1));
	assert(t.size(2));
	assert("TODO Tensor is float32");
	assert("TODO Tensor is on CUDA device");

	std::ofstream os(path, std::ofstream::binary);
	assert(os);

	uint32_t height = t.size(1);
	uint32_t width = t.size(2);
#if 0
	std::cout << "Packing " << path << " (" << width << "x" << height << ")" << std::endl;
#endif
	os << "M12R 0" << std::endl
	   << width << " " << height << std::endl
	   << 4095 << std::endl;

	uint8_t* packed_dev;
	uint8_t* packed_host;
	uint32_t packedSize = packedBufferSize(width, height);

	int rc = cudaMalloc(&packed_dev, packedSize);
	assert(cudaSuccess == rc);

	pack12(packed_dev, t.data<float>(), width, height, 0);

	rc = cudaHostAlloc(&packed_host, packedSize, cudaHostAllocMapped);
	assert(cudaSuccess == rc);

	rc = cudaMemcpy(packed_host, packed_dev, packedSize, cudaMemcpyDeviceToHost);
 	assert(cudaSuccess == rc);

	rc = cudaFree(packed_dev);
	assert(cudaSuccess == rc);

	os.write((char*)packed_host, packedSize);
	assert(os);
	os.close();

	rc = cudaFreeHost(packed_host);
	assert(cudaSuccess == rc);
}

torch::Tensor unpackM12(std::string path)
{
	std::ifstream is(path, std::ifstream::binary);
	assert(is);

	std::string marker;
	uint32_t rotation, width, height, _range;
	is >> marker >> rotation >> width >> height >> _range;
	is.get(); // newline after height

	uint32_t packedSize = packedBufferSize(width, height);
#if 0
	std::cout << "Unpacking " << path 
	     << " (" << width << "x" << height << ")"
	     << " [" << packedSize << "]" << std::endl;
#endif
	
	uint8_t* packed;
	float* unpacked;
	
	int rc = cudaHostAlloc(&packed, packedSize, cudaHostAllocMapped);
	assert(cudaSuccess == rc);

	rc = cudaMalloc(&unpacked, unpackedBufferSize<float>(width, height));
	assert(cudaSuccess == rc);

	is.read((char*)packed, packedSize);
	assert(is);
	is.close();

	unpack12(unpacked, packed, width, height, 0);
	
	rc = cudaStreamSynchronize(0);
	assert(cudaSuccess == rc);

	rc = cudaFreeHost(packed);
	assert(cudaSuccess == rc);
	
	auto options = torch::TensorOptions().dtype(torch::kFloat32). device(torch::kCUDA, 0);
	return torch::from_blob(unpacked, { 1, height, width }, { width * height, width, 1 }, cudaFree, options);
}

void compressB12(std::string path, torch::Tensor t, float threshold)
{
	assert(t.size(0) == 1);
	assert(t.size(1));
	assert(t.size(2));
	assert("TODO tensor internal type == F32");
	assert("TODO tensor on CUDA device");

	std::ofstream os(path, std::ofstream::binary);
	assert(os);
	
	uint32_t height = t.size(1);
	uint32_t width = t.size(2);
#if 0
	std::cout << "Compressing " << path << " (" << width << "x" << height << ")" << std::endl;
#endif	
	uint8_t* packed;
	float* copy;
	
	int rc = cudaMalloc(&copy, unpackedBufferSize<float>(width, height));
	assert(cudaSuccess == rc);
	
	rc = cudaMalloc(&packed, packedBufferSize(width, height));
	assert(cudaSuccess == rc);

	rc = cudaMemcpy(copy, t.data<float>(), width * height * sizeof(float), cudaMemcpyDeviceToDevice);
	assert(cudaSuccess == rc);

	Daubechies<2>::transform(copy, width, width, height, 0);
	Daubechies<2>::transform(copy, width, width, height, 0);
	Daubechies<4>::transformWithThreshold(copy, width, width, height, threshold, 0);
	pack12(packed, copy, width, height, 0);

	nvcomp::LZ4Manager m { 1<<16, NVCOMP_TYPE_CHAR, 0 };
	nvcomp::CompressionConfig config = m.configure_compression(packedBufferSize(width, height));

	uint8_t* compressed_dev;
	uint8_t* compressed_host;

	rc = cudaMalloc(&compressed_dev, config.max_compressed_buffer_size);
	assert(cudaSuccess == rc);
	
	m.compress(packed, compressed_dev, config);
	
	rc = cudaFree(packed);
	assert(cudaSuccess == rc);
	
	rc = cudaHostAlloc(&compressed_host, config.max_compressed_buffer_size, cudaHostAllocMapped);
	assert(cudaSuccess == rc);

	//TODO get_compressed_output_size should synchronize, which means the following 2 lines can be removed?
	//rc = cudaStreamSynchronize(0);
	//assert(cudaSuccess == rc);

	uint32_t compressedSize = m.get_compressed_output_size(compressed_dev);

	os << "B12R 0 " << width << " " << height << " " << compressedSize << std::endl;
	
	rc = cudaMemcpy(compressed_host, compressed_dev, compressedSize, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == rc);
	
	//rc = cudaStreamSynchronize(0);
	//assert(cudaSuccess == rc);

	rc = cudaFree(compressed_dev);
	assert(cudaSuccess == rc);

	os.write((char*)compressed_host, compressedSize);
	os.close();

	rc = cudaFreeHost(compressed_host);
	assert(cudaSuccess == rc);
}

void cudaFreeLogged(void* ptr)
{
	int rc = cudaFree(ptr);
	assert(cudaSuccess == rc);
}

torch::Tensor decompressB12(std::string path)
{
	std::ifstream is(path, std::ifstream::binary);
	assert(is);

	std::string marker;
	uint32_t rotation, width, height, compressedSize;
	is >> marker >> rotation >> width >> height >> compressedSize;
	is.get(); // newline after compressed size
#if 0
	std::cout << "Decompressing " << path 
	     << " (" << width << "x" << height << ")"
	     << " [" << compressedSize << "]" << std::endl;
#endif
	uint8_t* compressed;
	uint8_t* packed;
	float* unpacked;

	int rc = cudaMalloc(&packed, packedBufferSize(width, height));
	assert(cudaSuccess == rc);
	
	rc = cudaHostAlloc(&compressed, compressedSize, cudaHostAllocMapped);
	assert(cudaSuccess == rc);

	is.read((char*)compressed, compressedSize);
	assert(is);
	is.close();

	nvcomp::LZ4Manager m { 1<<16, NVCOMP_TYPE_UCHAR, 0 };
	nvcomp::DecompressionConfig config = m.configure_decompression(compressed);
	assert(packedBufferSize(width, height) == config.decomp_data_size);

	m.decompress(packed, compressed, config);
	
	rc = cudaFreeHost(compressed);
	assert(cudaSuccess == rc);
	
	rc = cudaMalloc(&unpacked, unpackedBufferSize<float>(width, height));
	assert(cudaSuccess == rc);
	
	unpack12(unpacked, packed, width, height, 0);
	
	rc = cudaFree(packed);
	assert(cudaSuccess == rc);
	
	Daubechies<4>::transformInv(unpacked, width, width, height, 0);
	Daubechies<2>::transformInv(unpacked, width, width, height, 0);
	Daubechies<2>::transformInv(unpacked, width, width, height, 0);
	
	auto options = torch::TensorOptions().dtype(torch::kFloat32). device(torch::kCUDA, 0);
	return torch::from_blob(unpacked, { 1, height, width }, { height * width, width, 1 }, cudaFreeLogged, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("shuffle", &shuffle, "Shuffle polar");
	m.def("shuffle4", &shuffle4, "Shuffle polar");
	m.def("unshuffle", &unshuffle, "Unshuffle polar");
	m.def("unshuffle4", &unshuffle4, "Unshuffle 2x2 to 4 planes");
	m.def("writeM12", &packM12, "Pack an M12 file");
	m.def("readM12", &unpackM12, "Unpack an M12 file");
	m.def("writeB12", &compressB12, "Compress tensor to a B12 file");
	m.def("readB12", &decompressB12, "Decompress a B12 file into a tensor");
	m.def("bayer", &bayer, "Apply bayer to RGB image to get single channel raw");
}
