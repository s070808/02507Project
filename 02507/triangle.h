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
		const float2 a;
		const float2 b;
		const float2 c;

		__device__ __host__ triangle(const float2 a, const float2 b, const float2 c)
			: a(a), b(b), c(c) {}

		__device__ __host__ float signed_area() const {
			return (
				x(a) * (y(b) - y(c)) +
				x(b) * (y(c) - y(a)) +
				x(c) * (y(a) - y(b))
				) * 0.5f;
		}
	};

	__device__ __host__ float signed_area(const float2 a, const float2 b, const float2 c) {
		return (
			x(a) * (y(b) - y(c)) +
			x(b) * (y(c) - y(a)) +
			x(c) * (y(a) - y(b))
			) * 0.5f;
	}
}