#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <vector>
#include <cmath>
#include <math.h>

#include <thrust/device_ptr.h>

#include "matrix.h"

#pragma once
namespace kp {
	using namespace thrust;

	__device__ __host__ float signed_area(const float2 &a, const float2 &b, const float2 &c) {
		return (
			x(a) * (y(b) - y(c)) +
			x(b) * (y(c) - y(a)) +
			x(c) * (y(a) - y(b))
			) * 0.5f;
	}

	class triangle
	{
	public:
		const float2 &a;
		const float2 &b;
		const float2 &c;

		__device__ __host__ triangle(const float2 &a, const float2 &b, const float2 &c)
			: a(a), b(b), c(c) {}

		__device__ __host__ float signed_area() const {
			return kp::signed_area(a, b, c);
		}
	};

	// CODE BELOW IS FROM ATTEMPT OF IMPLEMENTING TILE-BASED RENDERING

	struct bounding_box_in_coords {
		const device_ptr<float> vertices_x;
		const device_ptr<float> vertices_y;
		const device_ptr<unsigned int> triangles_a;
		const device_ptr<unsigned int> triangles_b;
		const device_ptr<unsigned int> triangles_c;
		const float min_x;
		const float max_x;
		const float min_y;
		const float max_y;

		bounding_box_in_coords(
			const device_ptr<float> vertices_x,
			const device_ptr<float> vertices_y,
			const device_ptr<unsigned int> triangles_a,
			const device_ptr<unsigned int> triangles_b,
			const device_ptr<unsigned int> triangles_c,
			const float min_x,
			const float max_x,
			const float min_y,
			const float max_y) :
			vertices_x(vertices_x),
			vertices_y(vertices_y),
			triangles_a(triangles_a),
			triangles_b(triangles_b),
			triangles_c(triangles_c),
			min_x(min_x),
			max_x(max_x),
			min_y(min_y),
			max_y(max_y) {}

		__host__ __device__ bool operator()(unsigned int i) {
			auto idx_a = triangles_a[i];
			auto idx_b = triangles_b[i];
			auto idx_c = triangles_c[i];

			auto a_x = vertices_x[idx_a];
			auto a_y = vertices_y[idx_a];
			auto b_x = vertices_x[idx_b];
			auto b_y = vertices_y[idx_b];
			auto c_x = vertices_x[idx_c];
			auto c_y = vertices_y[idx_c];

			return max(max(a_x, b_x), c_x) <= max_x
				&& min(min(a_x, b_x), c_x) >= min_x
				&& max(max(a_y, b_y), c_y) <= max_y
				&& min(min(a_y, b_y), c_y) >= min_y;
		}
	};

	std::vector<bounding_box_in_coords> create_bb_functors(const unsigned int n, const unsigned int m,
		const device_ptr<float> vertices_x,
		const device_ptr<float> vertices_y,
		const device_ptr<unsigned int> triangles_a,
		const device_ptr<unsigned int> triangles_b,
		const device_ptr<unsigned int> triangles_c) {

		std::vector<bounding_box_in_coords> functors;

		for (unsigned int i = 0; i < n; i++) {
			const float min_x = ((float)i / n) * 2.0f - 1.0f;
			const float max_x = ((float)(i + 1) / n) * 2.0f - 1.0f;
			for (unsigned int j = 0; j < m; j++) {
				const float min_y = ((float)i / n) * 2.0f - 1.0f;
				const float max_y = ((float)(i + 1) / n) * 2.0f - 1.0f;
				bounding_box_in_coords functor(vertices_x, vertices_y, triangles_a, triangles_b, triangles_c, min_x, max_x, min_y, max_y);
				functors.emplace_back(functor);
			}
		}

		return functors;
	}
}
