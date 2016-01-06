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

const int N = 32;
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

int main() {
	thrust::device_vector<float> screen_device(N*N);
	std::vector<float> triangle = {
		-0.6f, 1.0f,
		-1.0f, -0.8f,
		1.0f, -0.2f
	};
	thrust::device_vector<float> triangle_device = triangle;

	dim3 dimGrid(N / blocksize, N / blocksize);
	dim3 dimBlock(blocksize, blocksize);

	hello << <dimGrid, dimBlock >> >(screen_device.data(), triangle_device.data());

	thrust::host_vector<float> screen = screen_device;

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