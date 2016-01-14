#include "area_rasterizer.h"
#include "cuda_runtime.h"

area_rasterizer::area_rasterizer(const triangle triangle, const float inverse_area)
	: _inverse_area(inverse_area), _triangle(triangle) {}

__device__
glm::vec3 area_rasterizer::operator()(const glm::vec2 position) const {
	const triangle triangleA(_triangle.c, position, _triangle.b);
	const triangle triangleB(_triangle.a, position, _triangle.c);
	const triangle triangleC(_triangle.b, position, _triangle.a);

	const float areaA = triangleA.signed_area();
	const float areaB = triangleB.signed_area();
	const float areaC = triangleC.signed_area();

	if (areaA < 0.f || areaB < 0.f || areaC < 0.f) {
		return glm::vec3(0.f, 0.f, 0.f);
	}

	// Calculate barycentric coordinates
	return glm::vec3(
		areaA * _inverse_area,
		areaB * _inverse_area,
		areaC * _inverse_area
	);
}