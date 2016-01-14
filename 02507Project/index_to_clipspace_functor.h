#include "cuda_runtime.h"
#include <glm/glm.hpp>

#pragma once
class index_to_clipspace_functor
{
	const unsigned int _width;
	const unsigned int _height;
public:
	index_to_clipspace_functor(const unsigned int width, const unsigned int height);
	__device__ glm::vec2 operator()(const unsigned int index) const;
};

