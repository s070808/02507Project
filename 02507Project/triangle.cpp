#include "triangle.h"


triangle::triangle(const glm::vec2 a, const glm::vec2 b, const glm::vec2 c)
	: a(a), b(b), c(c) {}

float triangle::signed_area() const {
	return (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) * 0.5f;
}