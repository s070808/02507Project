#include <cmath>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "tiny_obj_loader.h"

#pragma once
namespace kp {
	struct std_scene {
		std::vector<float> vertices_x;
		std::vector<float> vertices_y;
		std::vector<float> vertices_z;
		std::vector<unsigned int> triangles_a;
		std::vector<unsigned int> triangles_b;
		std::vector<unsigned int> triangles_c;
	};

	float quadf(float value, float max) {
		return std::cosf(((value - (max / 2)) / (max / 2)) * M_PI) / 2 + 0.5f;
	}

	std_scene generate_cosine_scene(int imax, int jmax) {
		std_scene scene;

		for (int i = 0; i < imax; i++) {
			for (int j = 0; j < jmax; j++) {
				scene.vertices_x.push_back(((float)j / (jmax - 1)) * 2.f - 1.f);
				scene.vertices_y.push_back(((float)i / (imax - 1)) * (-2.f) + 1.f);
				scene.vertices_z.push_back(quadf(i, imax - 1) * quadf(j, jmax - 1));
			}
		}

		for (int i = 0; i < imax - 1; i++) {
			for (int j = 0; j < jmax - 1; j++) {
				scene.triangles_a.push_back(i*jmax + j);
				scene.triangles_b.push_back((i + 1)*jmax + j);
				scene.triangles_c.push_back(i*jmax + j + 1);

				scene.triangles_a.push_back((i + 1)*jmax + (j + 1));
				scene.triangles_b.push_back(i*jmax + j + 1);
				scene.triangles_c.push_back((i + 1)*jmax + j);
			}
		}

		return scene;
	}

	std_scene load_scene(std::string inputfile, float scaling_factor) {
		std::string err;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;

		if (!tinyobj::LoadObj(shapes, materials, err, inputfile.c_str())) {
			std::cerr << err << std::endl;
			exit(1);
		}

		std::cout << "# of shapes    : " << shapes.size() << std::endl;
		std::cout << "# of materials : " << materials.size() << std::endl;

		std_scene scene;
		float max_vertex_value = 1.0f;

		for (auto shape : shapes)
		for (auto position : shape.mesh.positions) {
			auto abs_pos = std::fabsf(position);
			if (abs_pos > max_vertex_value) {
				max_vertex_value = abs_pos;
			}
		}

		auto position_factor = scaling_factor / max_vertex_value;
		std::cout << "Max vertex value :" << max_vertex_value << std::endl;

		for (size_t i = 0; i < shapes.size(); i++) {
			auto shape = shapes[i];
			auto offset = scene.triangles_a.size();

			for (size_t j = 0; j < shape.mesh.indices.size(); j += 3) {
				scene.triangles_a.push_back(offset + shape.mesh.indices[j + 0]);
				scene.triangles_b.push_back(offset + shape.mesh.indices[j + 1]);
				scene.triangles_c.push_back(offset + shape.mesh.indices[j + 2]);
			}

			for (size_t j = 0; j < shape.mesh.positions.size(); j += 3) {
				scene.vertices_x.push_back(shape.mesh.positions[j + 0] * position_factor);
				scene.vertices_y.push_back(shape.mesh.positions[j + 1] * position_factor);
				scene.vertices_z.push_back(shape.mesh.positions[j + 2] * position_factor);
			}
		}

		return scene;
	}
}