#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "matrix.h"

#pragma once
namespace kp {
	using namespace thrust;

	class triangle
	{
	public:
		const Float2 a;
		const Float2 b;
		const Float2 c;

		__device__ __host__ triangle(const Float2 a, const Float2 b, const Float2 c)
			: a(a), b(b), c(c) {}

		__device__ __host__ float signed_area() const {
			return (
				x(a) * (y(b) - y(c)) +
				x(b) * (y(c) - y(a)) +
				x(c) * (y(a) - y(b))
				) * 0.5f;
		}
	};
}