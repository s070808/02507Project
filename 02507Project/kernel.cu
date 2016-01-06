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
void hello(device_ptr<char> a, device_ptr<int> b)
{
	// unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	// unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	a[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main()
{
	
	vector<char> a = { 'H', 'e', 'l', 'l', 'o', ' ' };
	vector<int> b = { 15, 10, 6, 0, -11, 1 };

	for (size_t i = 0; i < a.size(); i++) {
		std::cout << a[i];
	}

	device_vector<char> ad = a;
	device_vector<int> bd = b;

	dim3 dimGrid(N/blocksize, N/blocksize);
	dim3 dimBlock(blocksize, blocksize);
	
	hello<<<dimGrid, dimBlock>>>(ad.data(), bd.data());

	for (size_t i = 0; i < ad.size(); i++) {
		std::cout << ad[i];
	}

	std::cout << std::endl;
}