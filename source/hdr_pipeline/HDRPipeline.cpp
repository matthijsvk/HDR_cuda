


#include <framework/CUDA/error.h>

#include "HDRPipeline.h"


namespace
{
	template <typename T>
	auto cudaMalloc(std::size_t size)
	{
		void* ptr;
		throw_error(::cudaMalloc(&ptr, size * sizeof(T)));
		return cuda_unique_ptr<T> { static_cast<T*>(ptr) };
	}
}

void* downsample_buffer;
HDRPipeline::HDRPipeline(unsigned int width, unsigned int height)
	: d_input_image(cudaMalloc<float>(width * height * 3)),
	  d_luminance_image(cudaMalloc<float>(width * height)),
	  d_buffer_image(cudaMalloc<float>(width * height)),
	  d_brightpass_image(cudaMalloc<uchar4>(width * height)),
	  d_output_image(cudaMalloc<uchar4>(width * height)),
	  width(width),
	  height(height)
{

	// we need a memory buffer on the GPU to store the intermediate images we compute while downscaling -> CudaMalloc
	// we use floats, and only one color channel (luminance). Image is half size in width, and half in height -> 1/4 of original size

//	cudaError_t res = cudaMalloc(&downsample_buffer, width*height / 4.0f * sizeof(float));
//	if (res != cudaSuccess){
//		throw std::runtime_error("something went wrong with downscaling memory allocation");
//	}

	// this function does the same thing as the above, but a lot more concise
	throw_error(cudaMalloc(&downsample_buffer, width*height * sizeof(float) / 4));
}

void HDRPipeline::consume(const float* input_image)
{
	// copy input data to GPU
	throw_error(cudaMemcpy(d_input_image.get(), input_image, width * height * 3 * 4U, cudaMemcpyHostToDevice));
}

float HDRPipeline::downsample()
{
	// implement downsampling and return average luminance
	// call the function from hdr_pipeline.cu
	// dest and input buffers: see HDRPipeline declaration in header file
	void luminance(float* dest,	const float* input, unsigned int width, unsigned int height);
	luminance(d_luminance_image.get(),
			d_input_image.get(),
			width,
			height);

	void downsample(float* dest, float* input, unsigned int width, unsigned int height);
	downsample((float*)downsample_buffer, d_luminance_image.get(), width, height);

	//printf("average: %d ", *(float*)downsample_buffer);

	return 1.0;
}

void HDRPipeline::tonemap(float exposure, float brightpass_threshold)
{
	void tonemap(uchar4* tonemapped, uchar4* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold);

	tonemap(d_output_image.get(), d_brightpass_image.get(), d_input_image.get(), width, height, exposure, brightpass_threshold);

	// TODO: implement gaussian blur of brightpass
}

void HDRPipeline::compose()
{
	// TODO: compose tonemapped image with blurred brightpass
}


image<float> HDRPipeline::readLuminance()
{
	image<float> output(width, height);
	// copy back output data from GPU
	throw_error(cudaMemcpy(data(output), d_luminance_image.get(), width * height * 4U, cudaMemcpyDeviceToHost));
	return output;
}

image<float> HDRPipeline::readDownsample()
{
	image<float> output(width/2, height/2);
	// copy back output data from GPU
	throw_error(cudaMemcpy(data(output), downsample_buffer, width/2 * height/2 * 4U, cudaMemcpyDeviceToHost));
	return output;
}

image<std::uint32_t> HDRPipeline::readBrightpass()
{
	image<std::uint32_t> output(width, height);
	// copy back output data from GPU
	throw_error(cudaMemcpy(data(output), d_brightpass_image.get(), width * height * 4U, cudaMemcpyDeviceToHost));
	return output;
}

image<std::uint32_t> HDRPipeline::readOutput()
{
	image<std::uint32_t> output(width, height);
	// copy back output data from GPU
	throw_error(cudaMemcpy(data(output), d_output_image.get(), width * height * 4U, cudaMemcpyDeviceToHost));
	return output;
}
