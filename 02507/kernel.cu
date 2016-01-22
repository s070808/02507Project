#include "display.h"
#include "scene.h"
#include "render.h"

int main() {
	const int width = 1024, height = 1024;
	const int numVertices_x = 72, numVertices_y = 72;
	auto use_procedurally_generated_scene = true;

	auto image = new unsigned char[width*height * 3];
	auto scene = use_procedurally_generated_scene
		? kp::generate_cosine_scene(numVertices_x, numVertices_y)
		: kp::load_scene("scenes/teapot.obj", 1.0f);
	kp::render_image(image, width, height, scene, 0.25f);
	return kp::display_image(image, width, height);
}