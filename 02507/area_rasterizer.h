#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "matrix.h"
#include "triangle.h"

#pragma once
namespace kp {
	using namespace thrust;

	class area_rasterizer
	{
		const float _inverse_area;
		const triangle _triangle;
	public:
		__host__ __device__ area_rasterizer(const triangle triangle, const float inverse_area)
			: _inverse_area(inverse_area), _triangle(triangle) {}

		__host__ __device__ float3 operator()(const float2 position) const {
			const float areaA = signed_area(_triangle.c, position, _triangle.b);
			const float areaB = signed_area(_triangle.a, position, _triangle.c);
			const float areaC = signed_area(_triangle.b, position, _triangle.a);

			if (areaA < -(10e-6f) || areaB < -(10e-6f) || areaC < -(10e-6f)) {
				return make_tuple(0.f, 0.f, 0.f);
			}

			// Calculate barycentric coordinates
			return make_tuple(
				areaA * _inverse_area,
				areaB * _inverse_area,
				areaC * _inverse_area);
		}
	};
}
