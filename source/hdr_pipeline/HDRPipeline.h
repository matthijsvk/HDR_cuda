


#ifndef INCLUDED_HDRPIPELINE
#define INCLUDED_HDRPIPELINE

#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include <framework/image.h>


struct cudaFreeDeleter
{
	void operator()(void* ptr) const
	{
		cudaFree(ptr);
	}
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, cudaFreeDeleter>;


class HDRPipeline
{
	const unsigned int width;
	const unsigned int height;

	cuda_unique_ptr<float> d_input_image;
	cuda_unique_ptr<float> d_luminance_image;
	cuda_unique_ptr<float> d_buffer_image;
	cuda_unique_ptr<uchar4> d_brightpass_image;
	cuda_unique_ptr<uchar4> d_output_image;

public:
	HDRPipeline(unsigned int width, unsigned int height);

	void consume(const float* input_image);
	float downsample();
	void tonemap(float exposure, float brightpass_threshold);
	void compose();

	image<float> readLuminance();
	image<float> readDownsample();
	image<std::uint32_t> readBrightpass();
	image<std::uint32_t> readOutput();
};

#endif // INCLUDED_HDRPIPELINE
