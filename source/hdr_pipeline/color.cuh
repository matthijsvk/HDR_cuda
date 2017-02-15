


#include <math/vector.h>


__device__ float luminance(const math::float3& color)
{
	return dot(color, math::float3 { 0.2126f, 0.7152f, 0.0722f });
}

__device__ float Uncharted2Tonemap(float x)
{
	// from http://filmicworlds.com/blog/filmic-tonemapping-operators/
	constexpr float A = 0.15;
	constexpr float B = 0.50;
	constexpr float C = 0.10;
	constexpr float D = 0.20;
	constexpr float E = 0.02;
	constexpr float F = 0.30;
	return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
}

__device__ float tonemap(float c, float exposure)
{
	constexpr float W = 11.2;
	return Uncharted2Tonemap(c * exposure * 2.0f) / Uncharted2Tonemap(W);
}

__device__ math::float3 tonemap(const math::float3& c, float exposure)
{
	return { tonemap(c.x, exposure), tonemap(c.y, exposure), tonemap(c.z, exposure) };
}

__device__ unsigned char toLinear8(float c)
{
	return static_cast<unsigned char>(saturate(c) * 255.0f);
}

__device__ unsigned char toSRGB8(float c)
{
	return toLinear8(powf(c, 1.0f / 2.2f));
}

__device__ float fromLinear8(unsigned char c)
{
	return c * (1.0f / 255.0f);
}

__device__ float fromSRGB8(unsigned char c)
{
	return powf(fromLinear8(c), 2.2f);
}
