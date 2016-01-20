#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <thrust/tuple.h>

#pragma once

namespace kp {
	using namespace thrust;
	
	typedef unsigned int uint;
	typedef tuple<float, float> float2;
	typedef tuple<float, float, float> float3;
	typedef tuple<uint, uint, uint> uint3;

#pragma region x getters
	template<typename T>
	__device__ __host__ T inline x(const tuple<T, T> v) {
		return get<0>(v);
	}

	template<typename T>
	__device__ __host__ T inline x(const tuple<T, T, T> v) {
		return get<0>(v);
	}

	template<typename T>
	__device__ __host__ T inline x(const tuple<T, T, T, T> v) {
		return get<0>(v);
	}
#pragma endregion

#pragma region y getters
	template<typename T>
	__device__ __host__ T inline y(const tuple<T, T> v) {
		return get<1>(v);
	}

	template<typename T>
	__device__ __host__ T inline y(const tuple<T, T, T> v) {
		return get<1>(v);
	}

	template<typename T>
	__device__ __host__ T inline y(const tuple<T, T, T, T> v) {
		return get<1>(v);
	}
#pragma endregion

#pragma region z getters

	template<typename T>
	__device__ __host__ T inline z(const tuple<T, T, T> v) {
		return get<2>(v);
	}

	template<typename T>
	__device__ __host__ T inline z(const tuple<T, T, T, T> v) {
		return get<2>(v);
	}
#pragma endregion

#pragma region w getters

	template<typename T>
	__device__ __host__ T inline w(const tuple<T, T, T, T> v) {
		return get<3>(v);
	}
#pragma endregion

	__device__ __host__ float3 operator-(float3 u, float3 v) {
		return make_tuple(x(u) - x(v), y(u) - y(v), z(u) - z(v));
	}

	__device__ __host__ float3 operator/(float3 u, float3 v) {
		return make_tuple(x(u) / x(v), y(u) / y(v), z(u) / z(v));
	}
}
