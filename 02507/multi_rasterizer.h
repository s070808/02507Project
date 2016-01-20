#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <cuda.h>

#include <thrust/device_ptr.h>

#include "area_rasterizer.h"
#include "matrix.h"
#include "triangle.h"
#include "rasterize.h"

#pragma once
namespace kp {
	using namespace thrust;

	class multi_rasterizer {
		const unsigned int _n_triangles;
		const device_ptr<float> _vertices_x;
		const device_ptr<float> _vertices_y;
		const device_ptr<float> _vertices_z;
		const device_ptr<unsigned int> _triangles_a;
		const device_ptr<unsigned int> _triangles_b;
		const device_ptr<unsigned int> _triangles_c;
	public:
		__host__ __device__ multi_rasterizer(
			const unsigned int n_triangles,
			const device_ptr<float> vertices_x,
			const device_ptr<float> vertices_y,
			const device_ptr<float> vertices_z,
			const device_ptr<unsigned int> triangles_a,
			const device_ptr<unsigned int> triangles_b,
			const device_ptr<unsigned int> triangles_c) :
			_n_triangles(n_triangles),
			_vertices_x(vertices_x),
			_vertices_y(vertices_y),
			_vertices_z(vertices_z),
			_triangles_a(triangles_a),
			_triangles_b(triangles_b),
			_triangles_c(triangles_c){}

		__host__ __device__ tuple<float, float, float, float, int> operator()(const float2 position) const {
			// Use initial depth of less than -1.0f, as that is the maximum allowed.
			float maxDepth = -1.1f;
			auto result = make_tuple(0.0f, 0.0f, 0.0f, -1.0f, -1);

			for (unsigned int i = 0; i < _n_triangles; i++) {
				// Retrieve indices of triangle vertices
				auto idx_a = _triangles_a[i];
				auto idx_b = _triangles_b[i];
				auto idx_c = _triangles_c[i];

				// Retrieve X and Y components of triangle vertices 
				auto vertex_a = make_tuple(_vertices_x[idx_a], _vertices_y[idx_a]);
				auto vertex_b = make_tuple(_vertices_x[idx_b], _vertices_y[idx_b]);
				auto vertex_c = make_tuple(_vertices_x[idx_c], _vertices_y[idx_c]);

				// triangle t(vertex_a, vertex_b, vertex_c);
				// area_rasterizer rasterize(t, 1.0f / t.signed_area());
				float3 barycentric = rasterize_triangle(position, vertex_a, vertex_b, vertex_c);

				if (x(barycentric) + y(barycentric) + z(barycentric) > 10e-6f) {
					// We are inside triangle
					// Retrieve Z components of triangle vertices
					auto z_a = _vertices_z[idx_a];
					auto z_b = _vertices_z[idx_b];
					auto z_c = _vertices_z[idx_c];

					// Calculate interpolated depth
					auto depth = x(barycentric)*z_a + y(barycentric)*z_b + z(barycentric)*z_c;

					// Perform depth test
					if (depth > maxDepth) {
						maxDepth = depth;
						result = make_tuple(x(barycentric), y(barycentric), z(barycentric), depth, i);
					}
				}
			}

			// Return the foremost pixel value
			return result;
		}
	};
}