


#ifndef INCLUDED_CUDA_RUNTIME_ERROR
#define INCLUDED_CUDA_RUNTIME_ERROR

#pragma once

#include <exception>

#include <cuda_runtime_api.h>


namespace CUDA
{
	class error : public std::exception
	{
		cudaError err;

	public:
		error(cudaError err);

		const char* what() const noexcept;
	};

	inline void throw_error(cudaError err)
	{
		if (err != cudaSuccess)
			throw error(err);
	}
}

using CUDA::throw_error;

#endif  // INCLUDED_CUDA_RUNTIME_ERROR
