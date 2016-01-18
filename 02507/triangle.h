#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#pragma once
class triangle
{
public:
	const float2 a;
	const float2 b;
	const float2 c;

	__device__ __host__ triangle(const float2 a, const float2 b, const float2 c)
		: a(a), b(b), c(c) {}

	__device__ __host__ float signed_area() const {
		return (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) * 0.5f;
	}
};