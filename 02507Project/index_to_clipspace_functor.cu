#include "index_to_clipspace_functor.h"
#include "cuda_runtime.h"

index_to_clipspace_functor::index_to_clipspace_functor(const unsigned int width, const unsigned int height)
	: _width(width), _height(height) {}

__device__
glm::vec2 index_to_clipspace_functor::operator()(const unsigned int index) const {
	unsigned int pixel_x = index % _width;
	unsigned int pixel_y = index / _height;
	float screen_x = (float)pixel_x + 0.5f;
	float screen_y = (float)pixel_y + 0.5f;

	return glm::vec2(
		(screen_x * 2 - _width) / (_width - 1),
		-(screen_y * 2 - _height) / (_height - 1)
	);
}