


#include <math/vector.h>

#include "color.cuh"


namespace
{
	constexpr unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1) / b;
	}

	const math::float2 gauss_kernel[] = {
		math::float2(-16.00f, 0.00288204f),
		math::float2(-15.00f, 0.00418319f),
		math::float2(-14.00f, 0.00592754f),
		math::float2(-13.00f, 0.00819980f),
		math::float2(-12.00f, 0.01107369f),
		math::float2(-11.00f, 0.01459965f),
		math::float2(-10.00f, 0.01879116f),
		math::float2( -9.00f, 0.02361161f),
		math::float2( -8.00f, 0.02896398f),
		math::float2( -7.00f, 0.03468581f),
		math::float2( -6.00f, 0.04055144f),
		math::float2( -5.00f, 0.04628301f),
		math::float2( -4.00f, 0.05157007f),
		math::float2( -3.00f, 0.05609637f),
		math::float2( -2.00f, 0.05957069f),
		math::float2( -1.00f, 0.06175773f),
		math::float2(  0.00f, 0.06250444f),
		math::float2(  1.00f, 0.06175773f),
		math::float2(  2.00f, 0.05957069f),
		math::float2(  3.00f, 0.05609637f),
		math::float2(  4.00f, 0.05157007f),
		math::float2(  5.00f, 0.04628301f),
		math::float2(  6.00f, 0.04055144f),
		math::float2(  7.00f, 0.03468581f),
		math::float2(  8.00f, 0.02896398f),
		math::float2(  9.00f, 0.02361161f),
		math::float2( 10.00f, 0.01879116f),
		math::float2( 11.00f, 0.01459965f),
		math::float2( 12.00f, 0.01107369f),
		math::float2( 13.00f, 0.00819980f),
		math::float2( 14.00f, 0.00592754f),
		math::float2( 15.00f, 0.00418319f),
		math::float2( 16.00f, 0.00288204f)
	};
}

template<typename T>
void swap(T& a,T& b)
{
	T c = b;
	b = a;
	a = c;
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

__global__ void downsample_kernel(float* dest, const float* input, unsigned int width, unsigned int height){
	// each thread needs to know on which pixels to work -> get absolute coordinates of the thread in the grid
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		int F = 2; //width of downsampling square (F^2 = number of pixels to be pooled together)

		if ((x >= width / F ) || (y >= height / F))
			return;

		F = 4;

		float sum = 0.0f;

		// 2D version
//		for (int j = 0; j<F; j++){
//			for (int i = 0; i < F; i++){
//				// current pixel : (x+i, y+j)
//				sum += input[(y*F+j) * width + x*F +i];
//			}
//		}
//		dest[ y* width / F + x] = sum / (F * F);

		// 1D version : F is amount of pixels we pool
		for (int i = 0; i< F; i++){
			sum += input[F * (y * width + x) + i]; //jump with step pool_size
		}
		dest[y * width /F + x] = sum / F; //store the results. not * 3 b/c only luminance, no colors
}

// get the required number of blocks to cover the whole image, and run the kernel on all blocks
void downsample(float* dest,	const float* luminance, unsigned int width, unsigned int height){
	const dim3 block_size = { 32, 32 };
	// calculate number of blocks required to process the whole image -> round up to the next multiple of 32 (full block)
	const dim3 num_blocks = {
		divup(width, block_size.x),
		divup(height, block_size.y)
	};

	// launch the kernel that we wrote above for all the blocks
	downsample_kernel<<<num_blocks, block_size>>>(dest, luminance, width, height);
}


// TODO: implement gaussian blur for light bloom

__global__ void tonemap_kernel(uchar4* tonemapped, uchar4* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_thdesthold)
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
		uchar4 out = { toSRGB8(c_t.x), toSRGB8(c_t.y), toSRGB8(c_t.z), 0xFFU };
		tonemapped[y * width + x] = out;
		brightpass[y * width + x] = luminance(c_t) > brightpass_thdesthold ? out : uchar4 { 0U, 0U, 0U, 0xFFU };
	}
}

void tonemap(uchar4* tonemapped, uchar4* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_thdesthold)
{
	const auto block_size = dim3 { 32U, 32U };

	auto num_blocks = dim3{ divup(width, block_size.x), divup(height, block_size.y) };

	tonemap_kernel<<<num_blocks, block_size>>>(tonemapped, brightpass, src, width, height, exposure, brightpass_thdesthold);
}
