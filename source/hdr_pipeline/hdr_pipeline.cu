


#include <math/vector.h>

#include "color.cuh"


namespace
{
	constexpr unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1) / b;
	}

	template<typename T>
	void swap(T& a, T& b)
	{
		T c = b;
		b = a;
		a = c;
	}
}

// kernel that computes average luminance of a pixel
__global__ void luminance_kernel(float* dest, const float* input, unsigned int width, unsigned int height)
{
	// each thread needs to know on which pixels to work -> get absolute coordinates of the thread in the grid
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// input is stored as array (row-wise). Load the pixel values
	// offset of first pixel  = y coordinates * width of block (b/c full rows already read) + x coordinates

	if (x < width && y < height){
		const float* input_pixel = input + 3 * (width*y + x); // *3 b/c three colors (=bytes) per pixel.

		float lum =  0.21f * input_pixel[0] + 0.72f * input_pixel[1] + 0.07f * input_pixel[2];  //compensate for human vision

		dest[width *y + x] = lum; //store the results. not * 3 b/c only luminance, no colors
	}
}

// get the required number of blocks to cover the whole image, and run the kernel on all blocks
void luminance(float* dest,	const float* input, unsigned int width, unsigned int height){
	const dim3 block_size = { 32, 32 };
	// calculate number of blocks required to process the whole image -> round up to the next multiple of 32 (full block)
	const dim3 num_blocks = {
		divup(width, block_size.x),
		divup(height, block_size.y)
	};

	// launch the kernel that we wrote above for all the blocks
	luminance_kernel<<<num_blocks, block_size>>>(dest, input, width, height);

	// this 'downsampling' step takes about 1.3ms on a GT 730m
	// now we want to run this in a hierarchical way: reduce the image in each step
	// -> launch blocks on the images, reducing its size each step until we end up with a 1x1 color
	// -> we need to allocate memory for a buffer in the HDRPipeline object declaration
}

__global__ void downsample_kernel(float* dest, float* input, unsigned int width, unsigned int height, unsigned int outputPitch, unsigned int inputPitch){
	// each thread needs to know on which pixels to work -> get absolute coordinates of the thread in the grid
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		int F = 2; //width of downsampling square (F^2 = number of pixels to be pooled together)

		//printf(" KERNEL width %d | height %d \n", width, height);
		if ((x >= width / F ) || (y >= height / F))
			return;

		float sum = 0.0f;

		// 2D version: add pixels in 2x2 block, calculate the average. Jump with F so different threads don't operate on the same pixels.

		for (int j = 0; j<F; j++){
			for (int i = 0; i < F; i++){
				// current pixel : (x+i, y+j)
				sum += input[(y*F+j) * inputPitch + x*F +i];
			}
		}
		dest[ y* outputPitch + x] = sum / (F * F);
}

// get the required number of blocks to cover the whole image, and run the kernel on all blocks
float downsample(float* dest,	float* luminance, unsigned int width, unsigned int height){
	const dim3 block_size = { 32, 32 };
	// calculate number of blocks required to process the whole image -> round up to the next multiple of 32 (full block)
	const dim3 num_blocks = {
		divup(width, block_size.x),
		divup(height, block_size.y)
	};

	// Store original width = width of buffer
	const unsigned int pitchBuf = width /2;
	const unsigned int pitchLuminance = width;

	// first iteration
	downsample_kernel<<<num_blocks, block_size>>>(dest, luminance, width, height, pitchBuf, pitchLuminance);
	int ping = 0; //result in dest buffer

	while (width != 1 || height != 1){
		// result will be in the dest buffer
		width = width / 2;
		height = height / 2;
		if (width < 1){
			width = 1;
			printf("width < 1 \n");
		}
		if (height < 1){
			height = 1;
			printf("height < 1 \n");
		}

		printf(" width %d | height %d \n", width, height);
		if (ping){
			downsample_kernel<<<num_blocks, block_size>>>(dest, luminance, width, height, pitchBuf, pitchLuminance);
		}
		else {
			// now ping-pong; result will be in the luminance buffer
			downsample_kernel<<<num_blocks, block_size>>>(luminance, dest, width, height, pitchLuminance, pitchBuf);
		}
		ping = ! ping;
	}

	// make sure the result is stored in dest
	if (ping){
		cudaMemcpy(luminance, dest, 1, cudaMemcpyDeviceToDevice);
	}

	// return the grayscale value
	float average;
	cudaMemcpy(&average, dest, sizeof(float), cudaMemcpyDeviceToHost);
	return average;
}


__global__ void blur_kernel(float* dest, const float* src, unsigned int width, unsigned int height)
{
	constexpr float weights[] = {
		//  -16           -15         -14          -13          -12          -11          -10           -9           -8           -7           -6           -5           -4           -3           -2           -1
		0.00288204f, 0.00418319f, 0.00592754f, 0.00819980f, 0.01107369f, 0.01459965f, 0.01879116f, 0.02361161f, 0.02896398f, 0.03468581f, 0.04055144f, 0.04628301f, 0.05157007f, 0.05609637f, 0.05957069f, 0.06175773f,
		//    0
		0.06250444f,
		//    1             2           3            4            5            6            7            8            9           10           11           12           13           14           15           16
		0.06175773f, 0.05957069f, 0.05609637f, 0.05157007f, 0.04628301f, 0.04055144f, 0.03468581f, 0.02896398f, 0.02361161f, 0.01879116f, 0.01459965f, 0.01107369f, 0.00819980f, 0.00592754f, 0.00418319f, 0.00288204f
	};


}

void gaussian_blur(float* dest, const float* src, unsigned int width, unsigned int height)
{
	// TODO: gaussian blur of brightpass
}



void compose(float* output, const float* tonemapped, const float* blurred, unsigned int width, unsigned int height)
{
	// TODO: add blurred brightpass to tonemapped image
}



__global__ void tonemap_kernel(float* tonemapped, float* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		// figure out input color
		math::float3 c = { src[3 * (y * width + x) + 0], src[3 * (y * width + x) + 1], src[3 * (y * width + x) + 2] };

		// compute tonemapped color
		math::float3 c_t = tonemap(c, exposure);

		// write out tonemapped color
		tonemapped[3 * (y * width + x) + 0] = c_t.x;
		tonemapped[3 * (y * width + x) + 1] = c_t.y;
		tonemapped[3 * (y * width + x) + 2] = c_t.z;

		// write out brightpass color
		math::float3 c_b = luminance(c_t) > brightpass_threshold ? c_t : math::float3 {0.0f, 0.0f, 0.0f};
		brightpass[3 * (y * width + x) + 0] = c_b.x;
		brightpass[3 * (y * width + x) + 1] = c_b.y;
		brightpass[3 * (y * width + x) + 2] = c_b.z;
	}
}

void tonemap(float* tonemapped, float* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold)
{
	const auto block_size = dim3 { 32U, 32U };

	auto num_blocks = dim3{ divup(width, block_size.x), divup(height, block_size.y) };

	tonemap_kernel<<<num_blocks, block_size>>>(tonemapped, brightpass, src, width, height, exposure, brightpass_threshold);
}
