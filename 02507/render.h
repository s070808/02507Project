#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <ctime>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/sequence.h>

#include "index_to_clipspace_functor.h"
#include "multi_rasterizer.h"

#pragma once
namespace kp {
	using namespace thrust;

	struct rasterize_functor {
		const kp::index_to_clipspace_functor index_to_clipspace;
		const kp::multi_rasterizer rasterizer;

		__host__ __device__
			rasterize_functor(const kp::index_to_clipspace_functor index_to_clipspace, const kp::multi_rasterizer rasterizer)
			: index_to_clipspace(index_to_clipspace), rasterizer(rasterizer) {}

		__host__ __device__
			tuple<float, float, float, float, int> operator()(int i) {
				return rasterizer(index_to_clipspace(i));
			}
	};

	void render_image(unsigned char* image, int width, int height, kp::std_scene scene, float bd_scaling) {
		std::cout << "Number of triangles: " << scene.triangles_a.size() << std::endl;
		auto size = width * height;
		device_vector<float> screen_x(size), screen_y(size), screen_z(size), screen_depth(size);
		device_vector<int> screen_triangles(size);
		counting_iterator<int> begin(0);
		counting_iterator<int> end(size);
		device_vector<int> indices(size);
		sequence(indices.begin(), indices.end());

		auto t_begin = std::clock();

		// Copy vertices and triangles to GPU
		device_vector<float> vertices_x = scene.vertices_x;
		device_vector<float> vertices_y = scene.vertices_y;
		device_vector<float> vertices_z = scene.vertices_z;
		device_vector<unsigned int> triangles_a = scene.triangles_a;
		device_vector<unsigned int> triangles_b = scene.triangles_b;
		device_vector<unsigned int> triangles_c = scene.triangles_c;

		if (cudaDeviceSynchronize() != cudaSuccess) {
			std::cout << "CUDA error!" << std::endl;
			return;
		}

		const kp::index_to_clipspace_functor index_to_clipspace(width, height);
		const kp::multi_rasterizer rasterizer(
			triangles_a.size(),
			vertices_x.data(),
			vertices_y.data(),
			vertices_z.data(),
			triangles_a.data(),
			triangles_b.data(),
			triangles_c.data());

		auto screen_begin = make_tuple(screen_x.begin(), screen_y.begin(), screen_z.begin(), screen_depth.begin(), screen_triangles.begin());
		auto screen_end = make_tuple(screen_x.end(), screen_y.end(), screen_z.end(), screen_depth.end(), screen_triangles.end());

		transform(indices.begin(), indices.end(), make_zip_iterator(screen_begin), rasterize_functor(index_to_clipspace, rasterizer));

		if (cudaDeviceSynchronize() != cudaSuccess) {
			std::cout << "CUDA error!" << std::endl;
			return;
		}

		auto t_end = std::clock();
		auto elapsed_secs = double(t_end - t_begin) / CLOCKS_PER_SEC;
		std::cout << "Time elapsed: " << elapsed_secs*1000.0 << "ms" << std::endl;

		host_vector<float> host_x(size), host_y(size), host_z(size), host_depth(size);
		host_vector<int> host_triangles(size);
		auto host_begin = make_tuple(host_x.begin(), host_y.begin(), host_z.begin(), host_depth.begin(), host_triangles.begin());
		copy(make_zip_iterator(screen_begin), make_zip_iterator(screen_end), make_zip_iterator(host_begin));

		for (int i = 0; i < size; i++) {
			image[i * 3 + 0] = (unsigned char)((host_x[i]) * 255 * bd_scaling) +
				(unsigned char)((host_depth[i] * 0.5f + 0.5f) * 255 * (1.f - bd_scaling));

			image[i * 3 + 1] = (unsigned char)((host_y[i]) * 255 * bd_scaling) +
				(unsigned char)((host_depth[i] * 0.5f + 0.5f) * 255 * (1.f - bd_scaling));

			image[i * 3 + 2] = (unsigned char)((host_z[i]) * 255 * bd_scaling) +
				(unsigned char)((host_depth[i] * 0.5f + 0.5f) * 255 * (1.f - bd_scaling));
		}
	}
}