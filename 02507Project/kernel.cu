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
const int blocksize = 2;

__global__
void hello(thrust::device_ptr<float> screen, thrust::device_ptr<float> triangle) {
	unsigned int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;

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
		screen[N*pixel_y + pixel_x] = 1;
	}
	else {
		screen[N*pixel_y + pixel_x] = 0;
	}

	//screen[N*pixel_y + pixel_x] = point[1];

}

__device__
float rasterize_pixel(const unsigned int pixel_x, const unsigned int pixel_y, const thrust::device_ptr<float> triangle) {
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

int main() {
	thrust::host_vector<float> screen(N*N);
	thrust::device_vector<float> screen_d(N*N);
	std::vector<float> triangle = {
		-0.6f, 1.0f,
		-1.0f, -0.8f,
		1.0f, -0.2f
	};
	thrust::host_vector<unsigned int> indices(N*N);
	thrust::device_vector<float> triangle_d = triangle;
	thrust::device_ptr<float> triangle_data = triangle_d.data();

	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<unsigned int> indices_d = indices;
	std::cout << screen_d.size() << " " << indices_d.size() << std::endl;
	thrust::transform(indices_d.begin(), indices_d.end(), screen_d.begin(), [=] __device__(unsigned int i) {
		return rasterize_pixel(i%N, i/N, triangle_data);
	});

	std::cout << indices_d[12] << " " << indices_d[44] << std::endl;
	std::cout << screen_d[12] << " " << screen_d[44] << std::endl;

	screen = screen_d;


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
			float c = screen[y*N + x];// > 0 ? '#' : '.';
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