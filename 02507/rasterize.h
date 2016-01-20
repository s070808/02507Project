#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <cuda.h>

#include <thrust/tuple.h>
#include <thrust/device_ptr.h>

#include "area_rasterizer.h"
#include "matrix.h"
#include "triangle.h"

#pragma once
namespace kp {
	using namespace thrust;

	tuple<float, float, float> rasterize_triangle(float2 p, float2 a, float2 b, float2 c) {
		float2 v0 = b - a;
		float2 v1 = c - a;
		float2 v2 = p - a;
		float inv_den = 1.0f / (x(v0) * y(v1) - x(v1) * y(v0));
		float v = (x(v2) * y(v1) - x(v1) * y(v2)) * inv_den;

		if (v < 0.0f || v > 1.0f) {
			return make_tuple(0.0f, 0.0f, 0.0f);
		}

		float w = (x(v0) * y(v2) - x(v2) * y(v0)) * inv_den;

		if (w < 0.0f || w > 1.0f || v + w > 1.0f) {
			return make_tuple(0.0f, 0.0f, 0.0f);
		}

		float u = 1.0f - v - w;
		return make_tuple(u, v, w);
	}
}