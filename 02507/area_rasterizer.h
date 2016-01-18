#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "triangle.h"

#pragma once
namespace kp {
	using namespace thrust;

	class area_rasterizer
	{
		const float _inverse_area;
		const triangle _triangle;
	public:
		__device__ __host__ area_rasterizer(const triangle triangle, const float inverse_area)
			: _inverse_area(inverse_area), _triangle(triangle) {}

		__device__ __host__ tuple<float, float, float> operator()(const tuple<float, float> position) const {
			const triangle triangleA(_triangle.c, position, _triangle.b);
			const triangle triangleB(_triangle.a, position, _triangle.c);
			const triangle triangleC(_triangle.b, position, _triangle.a);

			const float areaA = triangleA.signed_area();
			const float areaB = triangleB.signed_area();
			const float areaC = triangleC.signed_area();

			if (areaA < 0.f || areaB < 0.f || areaC < 0.f) {
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
