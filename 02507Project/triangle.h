#include <glm/glm.hpp>

#pragma once
class triangle
{
public:
	const glm::vec2 a;
	const glm::vec2 b;
	const glm::vec2 c;
	triangle(const glm::vec2 a, const glm::vec2 b, const glm::vec2 c);
	float signed_area() const;
};

