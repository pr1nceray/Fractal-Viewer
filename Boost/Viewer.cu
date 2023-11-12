#include "Viewer.cuh"



Viewer::Viewer() {
	Set_Defaults();

	cudaError_t err;

	img_host = new sf::Uint8[sz_total];

	err = cudaMalloc((void**)&dest_dev, sz_total * sizeof(sf::Uint8));

	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc launch failed: %s\n", cudaGetErrorString(err));
		free_resources(dest_dev, img_host);		//MAYBE CHANGE?
		dest_dev = nullptr;
		img_host = nullptr;
	}

}

Viewer::~Viewer() {
	if (dest_dev != nullptr && img_host != nullptr) {
		free_resources(dest_dev, img_host);
	}
}
void Viewer::display() {
	cudaError_t err;
	while (window.isOpen()) {
		call_kernel<float>();
		Check_Events(); //adjust scaling
		sync();
		update_display();
	}
}

void Viewer::Set_Defaults() {
	res_x = 1024;
	res_y = 512;
	num_channels = 4;
	sz_total = res_x * res_y * num_channels;

	img_display.create(res_x, res_y);
	sprite = sf::Sprite(img_display);

	window.create(sf::VideoMode(1025, 512), "Fractal Viewer");
	//window.setFramerateLimit(60); // maybe remove?
	mode = 0;
}

void Viewer::Check_Events() {
	sf::Event e;

	while (window.pollEvent(e)) {
		if (e.type == sf::Event::Closed) {
			window.close();
		}
		if (e.type == sf::Event::MouseWheelMoved) {
			double moved = e.mouseWheel.delta;	//scaling needed here!
			if (moved > 0) {
				scale = scale / sqrt(abs(moved) + .025);
			}
			else {
				scale = scale * sqrt(abs(moved) + .025);
				scale = std::min(scale, (float)4.0);
			}
		}
	}
	Check_Keyboard();
}

void Viewer::Check_Keyboard() {
	int dir_y = (sf::Keyboard::isKeyPressed(sf::Keyboard::Up) ? 1 : 0) + (sf::Keyboard::isKeyPressed(sf::Keyboard::Down) ? -1 : 0);
	int dir_x = (sf::Keyboard::isKeyPressed(sf::Keyboard::Right) ? 1 : 0) + (sf::Keyboard::isKeyPressed(sf::Keyboard::Left) ? -1 : 0);
	center_y += dir_y * (scale / 64.0);
	center_x += dir_x * (scale / 64.0);
	if (abs(center_x) > 2) {
		center_x = 2 * (center_x < 0 ? -1 : 1);
	}
	if (abs(center_y) > 1) {
		center_y = 1 * (center_y < 0 ? -1 : 1);
	}
	std::cout << center_x << " " << center_y << "\n";
}

template<typename T>
void Viewer::call_kernel() {
	//Determine_ends<float> << <entire_block, xyblock >> > (dest_dev, scale, center_x, center_y, max_iters);

	switch (mode)
	{
	case 0: {
		Determine_ends<float> << <entire_block, xyblock >> > (dest_dev, scale, center_x, center_y, max_iters);
		break;
	}
	case 1: {
		//juliaSet
	}
	default:
		break;
	}
}

void Viewer::sync() {
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		free_resources(dest_dev, img_host);
	}
	cudaMemcpy(img_host, dest_dev, sz_total * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
}

void Viewer::update_display() {
	img_display.update(img_host);
	sprite.setTexture(img_display);
	window.clear();
	window.draw(sprite);
	window.display();
}

void Viewer::resize() {
	//update info	
}

void Viewer::free_resources(uint8_t* dest_dev, sf::Uint8* tmp) {
	cudaFree(dest_dev);
	delete[] tmp;

}
