#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#pragma once
class index_to_clipspace_functor
{
	const unsigned int _width;
	const unsigned int _height;
public:
	__device__ __host__ index_to_clipspace_functor(const unsigned int width, const unsigned int height)
		: _width(width), _height(height) {}

	__device__ __host__ float2 operator()(const unsigned int index) const {
		unsigned int pixel_x = index % _width;
		unsigned int pixel_y = index / _height;
		float screen_x = (float)pixel_x + 0.5f;
		float screen_y = (float)pixel_y + 0.5f;

		return make_float2(
			(screen_x * 2 - _width) / (_width - 1),
			-(screen_y * 2 - _height) / (_height - 1)
			);
	}
};

