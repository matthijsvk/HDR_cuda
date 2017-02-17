


#ifndef INCLUDED_HDRPIPELINE
#define INCLUDED_HDRPIPELINE

#pragma once

#include <memory>

#include <cuda_runtime_api.h>

#include <framework/rgb32f.h>
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
	cuda_unique_ptr<float> d_downsample_buffer;
	cuda_unique_ptr<float> d_tonemapped_image;
	cuda_unique_ptr<float> d_brightpass_image;
	cuda_unique_ptr<float> d_blurred_image;
	cuda_unique_ptr<float> d_output_image;

public:
	HDRPipeline(unsigned int width, unsigned int height);

	void consume(const float* input_image);
	void computeLuminance();
	float downsample();
	void tonemap(float exposure, float brightpass_threshold);
	void blur();
	void compose();

	image<float> readLuminance();
	image<float> readDownsample();
	image<RGB32F> readTonemapped();
	image<RGB32F> readBrightpass();
	image<RGB32F> readBlurred();
	image<RGB32F> readOutput();
};

#endif // INCLUDED_HDRPIPELINE
