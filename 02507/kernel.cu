#include "display.h"
#include "scene.h"
#include "render.h"

int main() {
	const int width = 512, height = 512;
	auto use_procedurally_generated_scene = false;
	auto image = new unsigned char[width*height * 3];
	auto scene = use_procedurally_generated_scene
		? kp::generate_cosine_scene(11, 11)
		: kp::load_scene("scenes/cube.obj", 1.0f);
	kp::render_image(image, width, height, scene, 0.25f);
	return kp::display_image(image, width, height);
}