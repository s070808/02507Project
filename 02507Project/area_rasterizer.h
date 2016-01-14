#include "cuda_runtime.h"
#include <glm/glm.hpp>
#include "triangle.h"

#pragma once
class area_rasterizer
{
	const float _inverse_area;
	const triangle _triangle;
public:
	area_rasterizer(const triangle triangle, const float inverse_area);
	__device__ glm::vec3 operator()(const glm::vec2 position) const;
};

