


#include <sstream>

#include "error.h"


namespace CUDA
{
	error::error(cudaError err)
		: err(err)
	{
	}

	const char* error::what() const noexcept
	{
		return cudaGetErrorString(err);
	}
}
