#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <vector>
#include <iostream>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

__device__ const int N = 128;

struct rasterize_functor {
	const unsigned int N;
	const thrust::device_ptr<float> triangle;

	rasterize_functor(const unsigned int N, const thrust::device_ptr<float> triangle) : N(N), triangle(triangle) {}

	__device__
		float operator()(const unsigned int& i) const {
			unsigned int pixel_x = i % N;
			unsigned int pixel_y = i / N;
			float screen_x = (float)pixel_x + 0.5f;
			float screen_y = (float)pixel_y + 0.5f;

			float point[2] = {
				(screen_x * 2 - N) / (N - 1),
				-(screen_y * 2 - N) / (N - 1)
			};

			float A = 1.f / 2.f * (-triangle[3] * triangle[4] + triangle[1] * (-triangle[2] + triangle[4]) + triangle[0] * (triangle[3] - triangle[5]) + triangle[2] * triangle[5]);
			float sign = A < 0 ? -1 : 1;
			float s = (triangle[1] * triangle[4] - triangle[0] * triangle[5] + (triangle[5] - triangle[1]) * point[0] + (triangle[0] - triangle[4]) * point[1]) * sign;
			float t = (triangle[0] * triangle[3] - triangle[1] * triangle[2] + (triangle[1] - triangle[3]) * point[0] + (triangle[2] - triangle[0]) * point[1]) * sign;

			bool in_triangle = s >= 0 && t >= 0 && ((s + t) <= 2 * A * sign);

			if (in_triangle) {
				return 1.f;
			}
			else {
				return 0.f;
			}
		}
};

void display_pixels(thrust::host_vector<float> screen) {
	std::cout << "+-";
	for (size_t i = 0; i < N; i++)
	{
		std::cout << "--";
	}
	std::cout << "+";
	std::cout << std::endl;
	for (size_t y = 0; y < N; y++) {
		std::cout << "| ";
		for (size_t x = 0; x < N; x++) {
			char c = screen[y*N + x] > 0 ? '#' : '.';
			std::cout << c << " ";
		}
		std::cout << "|";
		std::cout << std::endl;
	}
	std::cout << "+-";
	for (size_t i = 0; i < N; i++)
	{
		std::cout << "--";
	}
	std::cout << "+";
	std::cout << std::endl;
}

GLubyte screenData[N][N][3];
GLuint gl_texturePtr = 0;
uchar4* h_textureBufferData = nullptr;
uchar4* d_textureBufferData = nullptr;

void initBuffers(int width, int height) {
	// Only needed if we were to initialize buffers multiple times, which we don't do currently.. ¯\_(ツ)_/¯
	delete[] h_textureBufferData;
	h_textureBufferData = nullptr;
	glDeleteTextures(1, &gl_texturePtr);
	gl_texturePtr = 0;

	h_textureBufferData = new uchar4[width * height];

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_texturePtr);
	glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_textureBufferData);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void setupTexture(int width, int height) {
	// Clear screen
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			screenData[y][x][0] = screenData[y][x][1] = screenData[y][x][2] = 0;
		}
	}

	// Create a texture 
	glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)screenData);

	// Set up the texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Enable textures
	glEnable(GL_TEXTURE_2D);
}

void updateTexture(thrust::host_vector<float>& screen) {
	for (size_t y = 0; y < N; y++) {
		for (size_t x = 0; x < N; x++) {
			float pixelf = screen[y*N + x] * 125;
			screenData[y][x][0] = (GLubyte)230;
			screenData[y][x][1] = (GLubyte)230;
			screenData[y][x][2] = (GLubyte)230;
			std::cout << screenData[y][x][1] << "-";
		}
	}
}

void drawTexture(int width, int height) {
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)screenData);

	glBegin(GL_QUADS);
	glTexCoord2d(0.0, 0.0);		glVertex2d(0.0, 0.0);
	glTexCoord2d(1.0, 0.0); 	glVertex2d(width, 0.0);
	glTexCoord2d(1.0, 1.0); 	glVertex2d(width, height);
	glTexCoord2d(0.0, 1.0); 	glVertex2d(0.0, height);
	glEnd();
}

int main() {

	thrust::device_vector<float> screen_d(N*N);

	// Initialize a triangle from three points
	std::vector<float> triangle = {
		-0.6f, 1.0f,
		-1.0f, -0.8f,
		1.0f, -0.2f
	};

	thrust::device_vector<float> triangle_d = triangle;
	thrust::device_ptr<float> triangle_data = triangle_d.data();

	thrust::device_vector<unsigned int> indices(N*N);
	thrust::sequence(indices.begin(), indices.end());
	thrust::transform(indices.begin(), indices.end(), screen_d.begin(), rasterize_functor(N, triangle_data));

	thrust::host_vector<float> screen = screen_d;

	// display_pixels(screen);

	// Initialize GLFW
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a OpenGL window
	GLFWwindow* window = glfwCreateWindow(N, N, "LearnOpenGL", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize OpenGL functions with GLEW
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return -1;
	}

	// Set OpenGL viewport
	glViewport(0, 0, N, N);
	glClearColor(0.2, 0.2, 0.2, 1.0);

	float* pixels = new float[N * N * 3];

	for (size_t y = 0; y < N; y++) {
		for (size_t x = 0; x < N; x++) {
			float pixelf = screen[y*N + x]*255;
			pixels[y*N + x + 0] = pixelf;
			pixels[y*N + x + 1] = pixelf;
			pixels[y*N + x + 2] = pixelf;
			std::cout << pixels[y*N + x + 2] << "-";
		}
	}

	//setupTexture(N, N);

	// Start render loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDrawPixels(N, N, GL_RGB, GL_FLOAT, pixels);
		//glRasterPos2i(0, 0);
		//glDrawPixels(N, N, GL_RGB, GL_UNSIGNED_BYTE, screenData);
		// glFlush();

		glfwSwapBuffers(window);
	}

	return 0;
}