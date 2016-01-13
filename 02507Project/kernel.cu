#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <vector>
#include <iostream>
#include <ctime>
#define _USE_MATH_DEFINES
#include <math.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Shader.h"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include "triangle.h"
#include "area_rasterizer.h"

__device__ const int N = 512;
const unsigned int WIDTH = N, HEIGHT = N;

struct rasterize_functor {
	const unsigned int width;
	const unsigned int height;
	const thrust::device_ptr<float> triangle;

	rasterize_functor(const unsigned int width, const unsigned int height, const thrust::device_ptr<float> triangle)
		: width(width), height(height), triangle(triangle) {}

	__device__
		float3 operator()(const unsigned int& i) const {
			unsigned int pixel_x = i % width;
			unsigned int pixel_y = i / height;
			float screen_x = (float)pixel_x + 0.5f;
			float screen_y = (float)pixel_y + 0.5f;

			float point[2] = {
				(screen_x * 2 - width) / (width - 1),
				-(screen_y * 2 - height) / (height - 1)
			};

			float A = 1.f / 2.f * (-triangle[3] * triangle[4] + triangle[1] * (-triangle[2] + triangle[4]) + triangle[0] * (triangle[3] - triangle[5]) + triangle[2] * triangle[5]);
			float sign = A < 0 ? -1 : 1;
			float s = (triangle[1] * triangle[4] - triangle[0] * triangle[5] + (triangle[5] - triangle[1]) * point[0] + (triangle[0] - triangle[4]) * point[1]) * sign;
			float t = (triangle[0] * triangle[3] - triangle[1] * triangle[2] + (triangle[1] - triangle[3]) * point[0] + (triangle[2] - triangle[0]) * point[1]) * sign;

			bool in_triangle = s >= 0 && t >= 0 && ((s + t) <= 2 * A * sign);
			float3 coords;

			if (in_triangle) {
				s = s / (A*2);
				t = t / (A*2);
				coords.x = s;
				coords.y = t;
				coords.z = 1.0f - s - t;
			}
			else {
				coords.x = 0.0f;
				coords.y = 0.0f;
				coords.z = 0.0f;
			}

			return coords;
		}
};

void generate_image(unsigned char* image) {
	thrust::device_vector<float3> screen_d(WIDTH*HEIGHT);

	// Initialize a triangle from three points
	std::vector<float> triangle = {
		-0.6f, 1.0f,
		-1.0f, -0.8f,
		1.0f, -0.2f
	};

	thrust::device_vector<float> triangle_d = triangle;
	thrust::device_ptr<float> triangle_data = triangle_d.data();

	thrust::device_vector<unsigned int> indices(WIDTH*HEIGHT);
	thrust::sequence(indices.begin(), indices.end());
	thrust::transform(indices.begin(), indices.end(), screen_d.begin(), rasterize_functor(WIDTH, HEIGHT, triangle_data));

	thrust::host_vector<float3> screen = screen_d;
	
	for (int i = 0; i < screen.size(); i++) {
		float3 value = screen[i];
		image[i * 3 + 0] = value.x * 255;
		image[i * 3 + 1] = value.y * 255;
		image[i * 3 + 2] = value.z * 255;
	}
}

void generate_image2(unsigned char* image) {
	thrust::device_vector<float3> screen_d(WIDTH*HEIGHT);
	thrust::device_vector<unsigned int> indices(WIDTH*HEIGHT);
	thrust::sequence(indices.begin(), indices.end());

	triangle t(glm::vec2(1.0f, 0.25f), glm::vec2(0.66f, 1.0f), glm::vec2(0.0f, 0.33f));
	thrust::transform(indices.begin(), indices.end(), screen_d.begin(), rasterize_functor(WIDTH, HEIGHT, triangle_data));

	thrust::host_vector<float3> screen = screen_d;

	for (int i = 0; i < screen.size(); i++) {
		float3 value = screen[i];
		image[i * 3 + 0] = value.x * 255;
		image[i * 3 + 1] = value.y * 255;
		image[i * 3 + 2] = value.z * 255;
	}
}

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// Window dimensions

int main() {
	triangle t(glm::vec2(1.0f, 0.25f), glm::vec2(0.66f, 1.0f), glm::vec2(0.0f, 0.33f));
	area_rasterizer rasterizer(t, 1.0f/t.signed_area());
	auto bc = rasterizer(glm::vec2(1.0f, 0.5f));
	std::cout << t.signed_area() << std::endl;
	std::cout << bc.x << " " << bc.y << " " << bc.z << std::endl;

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
	clock_t begin, end;
	double elapsed_secs;

	//begin = std::clock();
	//generate_image(image);
	//end = std::clock();
	//elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	//std::cout << "First run: " << elapsed_secs*1000.0 << "ms" << std::endl;

	//begin = std::clock();
	//generate_image(image);
	//end = std::clock();
	//elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	//std::cout << "Second run: " << elapsed_secs*1000.0 << "ms" << std::endl;

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