#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <vector>
#include <iostream>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

__device__ const int N = 32;

struct rasterize_functor {
	const unsigned int N;
	const thrust::device_ptr<float> triangle;

	rasterize_functor(const unsigned int N, const thrust::device_ptr<float> triangle) : N(N), triangle(triangle) {}

	__device__
		float operator()(const unsigned int& i) const {
			unsigned int pixel_x = i % N;
			unsigned int pixel_y = i / N;
			float screen_x = (float)pixel_x + 0.5f;
			float screen_y = (float)pixel_y + 0.5f;

			float point[2] = {
				(screen_x * 2 - N) / (N - 1),
				-(screen_y * 2 - N) / (N - 1)
			};

			float A = 1.f / 2.f * (-triangle[3] * triangle[4] + triangle[1] * (-triangle[2] + triangle[4]) + triangle[0] * (triangle[3] - triangle[5]) + triangle[2] * triangle[5]);
			float sign = A < 0 ? -1 : 1;
			float s = (triangle[1] * triangle[4] - triangle[0] * triangle[5] + (triangle[5] - triangle[1]) * point[0] + (triangle[0] - triangle[4]) * point[1]) * sign;
			float t = (triangle[0] * triangle[3] - triangle[1] * triangle[2] + (triangle[1] - triangle[3]) * point[0] + (triangle[2] - triangle[0]) * point[1]) * sign;

			bool in_triangle = s >= 0 && t >= 0 && ((s + t) <= 2 * A * sign);

			if (in_triangle) {
				return 1.f;
			}
			else {
				return 0.f;
			}
		}
};

void display_pixels(thrust::host_vector<float> screen) {
	std::cout << "+-";
	for (size_t i = 0; i < N; i++)
	{
		std::cout << "--";
	}
	std::cout << "+";
	std::cout << std::endl;
	for (size_t y = 0; y < N; y++) {
		std::cout << "| ";
		for (size_t x = 0; x < N; x++) {
			char c = screen[y*N + x] > 0 ? '#' : '.';
			std::cout << c << " ";
		}
		std::cout << "|";
		std::cout << std::endl;
	}
	std::cout << "+-";
	for (size_t i = 0; i < N; i++)
	{
		std::cout << "--";
	}
	std::cout << "+";
	std::cout << std::endl;
}

int main() {
	thrust::device_vector<float> screen_d(N*N);
	std::vector<float> triangle = {
		-0.6f, 1.0f,
		-1.0f, -0.8f,
		1.0f, -0.2f
	};
	thrust::device_vector<float> triangle_d = triangle;
	thrust::device_ptr<float> triangle_data = triangle_d.data();

	thrust::device_vector<unsigned int> indices(N*N);
	thrust::sequence(indices.begin(), indices.end());
	thrust::transform(indices.begin(), indices.end(), screen_d.begin(), rasterize_functor(N, triangle_data));

	thrust::host_vector<float> screen = screen_d;

	display_pixels(screen);

	return 0;
}