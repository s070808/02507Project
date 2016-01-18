#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#pragma once
namespace kp {
	using namespace thrust;

	class triangle
	{
	public:
		const tuple<float, float> a;
		const tuple<float, float> b;
		const tuple<float, float> c;

		__device__ __host__ triangle(const tuple<float, float> a, const tuple<float, float> b, const tuple<float, float> c)
			: a(a), b(b), c(c) {}

		__device__ __host__ float signed_area() const {
			return (
				get<0>(a) * (get<1>(b) -get<1>(c)) +
				get<0>(b) * (get<1>(c) -get<1>(a)) +
				get<0>(c) * (get<1>(a) -get<1>(b))
				) * 0.5f;
		}
	};
}