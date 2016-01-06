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

using namespace std;
using namespace thrust;

const int N = 32;
const int blocksize = 16;

__global__
void hello(device_ptr<char> screen, device_ptr<float> triangle) {
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	screen[N*y + x] = 48 + x;
}

int main() {
	device_vector<char> screen(N*N);
	vector<float> triangle = {
		-1.0, -1.0,
		1.0, -1.0,
		1.0, 1.0
	};
	device_vector<float> triangle_device = triangle;

	dim3 dimGrid(N/blocksize, N/blocksize);
	dim3 dimBlock(blocksize, blocksize);
	
	hello<<<dimGrid, dimBlock>>>(screen.data(), triangle_device.data());

	for (size_t y = 0; y < N; y++) {
		for (size_t x = 0; x < N; x++) {
			std::cout << screen[y*N + x];
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}