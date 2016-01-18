#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#pragma once
namespace kp {
	using namespace thrust;

	class index_to_clipspace_functor
	{
		const int _width;
		const int _height;
	public:
		__device__ __host__ index_to_clipspace_functor(const int width, const int height)
			: _width(width), _height(height) {}

		__device__ __host__ tuple<float, float> operator()(const int index) const {
			int pixel_x = index % _width;
			int pixel_y = index / _height;
			float screen_x = (float)pixel_x + 0.5f;
			float screen_y = (float)pixel_y + 0.5f;

			return make_tuple(
				(screen_x * 2 - _width) / (_width - 1),
				(screen_y * 2 - _height) / (_height - 1)
				);
		}
	};
}