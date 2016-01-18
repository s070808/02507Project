#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <vector>
#include <iostream>
#include <ctime>
#define _USE_MATH_DEFINES
#include <math.h>

#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>
#include "Shader.h"

#include "matrix.h"
#include "triangle.h"
#include "multi_rasterizer.h"
#include "index_to_clipspace_functor.h"

using namespace thrust;

const int N = 512;
const int WIDTH = N, HEIGHT = N;

namespace kp {
	struct rasterize_functor {
		const kp::index_to_clipspace_functor index_to_clipspace;
		const kp::multi_rasterizer rasterizer;

		__host__ __device__ rasterize_functor(const kp::index_to_clipspace_functor index_to_clipspace, const kp::multi_rasterizer rasterizer)
			: index_to_clipspace(index_to_clipspace), rasterizer(rasterizer) {}

		__host__ __device__ tuple<float, float, float, uint> operator()(int i) {
			return rasterizer(index_to_clipspace(i));
		}
	};

	__host__ void generate_image2(unsigned char* image) {
		auto size = WIDTH * HEIGHT;

		// To be used later for optimization
		counting_iterator<int> begin(0);
		counting_iterator<int> end(size);

		device_vector<float> screen_x(size), screen_y(size), screen_z(size);
		device_vector<uint> screen_triangle(size);
		device_vector<int> indices(size);
		sequence(indices.begin(), indices.end());

		auto t_begin = std::clock();

		std::vector<float> std_vertices_x{ -1.0f, 0.66f, 0.0f, 1.0f, -0.75f, 0.0f, -1.0f, 1.0f };
		std::vector<float> std_vertices_y{ -0.75f, -1.0f, 1.0f, 1.0f, 0.75f, -1.0f, 0.0f, 0.0f };

		// Indices for corners A, B and C of triangles to be rasterized
		std::vector<unsigned int> std_triangles_a{ 5, 0, 2, 0 };
		std::vector<unsigned int> std_triangles_b{ 7, 1, 1, 2 };
		std::vector<unsigned int> std_triangles_c{ 6, 2, 3, 4 };

		// Copy vertices and triangles to GPU
		device_vector<float> vertices_x = std_vertices_x;
		device_vector<float> vertices_y = std_vertices_y;
		device_vector<unsigned int> triangles_a = std_triangles_a;
		device_vector<unsigned int> triangles_b = std_triangles_b;
		device_vector<unsigned int> triangles_c = std_triangles_c;

		const kp::index_to_clipspace_functor index_to_clipspace(WIDTH, HEIGHT);
		const kp::multi_rasterizer rasterizer(
			triangles_a.size(),
			vertices_x.data(),
			vertices_y.data(),
			triangles_a.data(),
			triangles_b.data(),
			triangles_c.data());

		transform(indices.begin(), indices.end(),
			make_zip_iterator(make_tuple(screen_x.begin(), screen_y.begin(), screen_z.begin(), screen_triangle.begin())),
			rasterize_functor(index_to_clipspace, rasterizer));

		auto t_end = std::clock();
		auto elapsed_secs = double(t_end - t_begin) / CLOCKS_PER_SEC;
		std::cout << "Time elapsed: " << elapsed_secs*1000.0 << "ms  (" << (t_end - t_begin) << " clocks)" << std::endl;

		host_vector<float> host_x(size), host_y(size), host_z(size);
		host_vector<uint> host_triangle(size);
		copy(
			make_zip_iterator(make_tuple(screen_x.begin(), screen_y.begin(), screen_z.begin(), screen_triangle.begin())),
			make_zip_iterator(make_tuple(screen_x.end(), screen_y.end(), screen_z.end(), screen_triangle.end())),
			make_zip_iterator(make_tuple(host_x.begin(), host_y.begin(), host_z.begin(), host_triangle.begin()))
			);

		for (int i = 0; i < size; i++) {
			image[i * 3 + 0] = (unsigned char)(screen_triangle[i] * 63);
			image[i * 3 + 1] = (unsigned char)(screen_triangle[i] * 63);
			image[i * 3 + 2] = (unsigned char)(screen_triangle[i] * 63);
		}
	}
}

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// Window dimensions

int main() {
	// Init GLFW
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Set the required callback functions
	glfwSetKeyCallback(window, key_callback);

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, WIDTH, HEIGHT);


	// Build and compile our shader program
	Shader ourShader("textures.vert", "textures.frag");


	// Set up vertex data (and buffer(s)) and attribute pointers
	GLfloat vertices[] = {
		// Positions          // Colors           // Texture Coords
		1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // Top Right
		1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // Bottom Right
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // Bottom Left
		-1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f  // Top Left 
	};
	GLuint indices[] = {  // Note that we start from 0!
		0, 1, 3, // First Triangle
		1, 2, 3  // Second Triangle
	};
	GLuint VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
	// TexCoord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0); // Unbind VAO


	// Load and create a texture 
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture); // All upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// Set the texture wrapping parameters
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// Set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Load image, create texture
	int widths = WIDTH;
	int heights = HEIGHT;
	unsigned char* image = new unsigned char[widths*heights * 3];

	kp::generate_image2(image);
	//kp::generate_image2(image);
	//kp::generate_image2(image);
	//kp::generate_image2(image);
	//kp::generate_image2(image);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, widths, heights, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	//glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done, so we won't accidentily mess up our texture.


	// Game loop
	while (!glfwWindowShouldClose(window))
	{
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();

		// Render
		// Clear the colorbuffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);


		// Bind Texture
		glBindTexture(GL_TEXTURE_2D, texture);

		// Activate shader
		ourShader.Use();

		// Draw container
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// Swap the screen buffers
		glfwSwapBuffers(window);
	}
	// Properly de-allocate all resources once they've outlived their purpose
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();
	return 0;
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}