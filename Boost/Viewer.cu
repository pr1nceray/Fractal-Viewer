#include "Viewer.cuh"



Viewer::Viewer() {
	Set_Defaults();

	cudaError_t err;

	img_host = new sf::Uint8[sz_max];

	err = cudaMalloc((void**)&dest_dev, sz_max * sizeof(sf::Uint8));

	Cuda_Verify("cudaMalloc launch failed:", err);

}

Viewer::~Viewer() {
	if (dest_dev != nullptr && img_host != nullptr) {
		free_resources(dest_dev, img_host);
	}
}
void Viewer::display() {
	try {
		while (window.isOpen()) {
			if (!precise) {
				call_kernel<float>();
			}
			else {
				call_kernel<double>();
			}
			Check_Events(); //adjust scaling
			sync();
			update_display();
		}
	}
	catch (CudaException& e) {
		std::cout << e.what() << " ; error code : " << e.code;
		free_resources(dest_dev, img_host);
	}

}

void Viewer::Init_Menu() {
	//setup menu for the viewer
}

void Viewer::Set_Defaults() {
	res.x = 1024;
	res.y = 512;
	aspect_ratio = 2;

	num_channels = 4;
	sz_total = 1024 * 512 * num_channels;

	img_display.create(res.x, res.y);
	sprite = sf::Sprite(img_display);

	window.create(sf::VideoMode(1024, 512), "Fractal Viewer");
	window.setFramerateLimit(60); // maybe remove?
	mode = 3;

	last_mouse = { -1, -1 };

	center.x = 0;
	center.y = 0;
}

void Viewer::Check_Events() {
	sf::Event e;

	while (window.pollEvent(e)) {
		if (e.type == sf::Event::Closed) {
			window.close();
		}
		if (e.type == sf::Event::MouseWheelMoved) {
			float moved = e.mouseWheel.delta;	//scaling needed here!
			if (moved > 0) {
				scale = scale / sqrt(abs(moved) + .025);
			}
			else {
				scale = scale * sqrt(abs(moved) + .025);
				scale = std::min(scale, 4.0);
			}
		}
		if (e.type == sf::Event::Resized) {
			resize();
		}
	}

	Check_Keyboard();
	Check_Mouse();
	if (abs(center.x) > 2) {
		center.x = 2 * (center.x < 0 ? -1 : 1);
	}
	if (abs(center.y) > 1) {
		center.y = 1 * (center.y < 0 ? -1 : 1);
	}

}

void Viewer::Check_Keyboard() {
	int dir_y = (sf::Keyboard::isKeyPressed(sf::Keyboard::Up) ? 1 : 0) + (sf::Keyboard::isKeyPressed(sf::Keyboard::Down) ? -1 : 0);
	int dir_x = (sf::Keyboard::isKeyPressed(sf::Keyboard::Right) ? 1 : 0) + (sf::Keyboard::isKeyPressed(sf::Keyboard::Left) ? -1 : 0);
	center.y += dir_y * (scale / 64.0);
	center.x += dir_x * (scale / 64.0);
	
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
		stbi_write_png("screenshot.png", res.x, res.y, num_channels, img_host, num_channels * res.x);
	}
}

void Viewer::Check_Mouse() {
	if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
		auto mouse_pos = sf::Mouse::getPosition(window);

		if (last_mouse.x != -1) {
			//multiply by scale 
			center.x -= ((mouse_pos.x - last_mouse.x) / (float)res.x) * scale * aspect_ratio; 
			center.y += ((mouse_pos.y - last_mouse.y) / (float)res.y) * scale;
		}
		last_mouse = mouse_pos;
	}
	else {
		last_mouse.x = -1; last_mouse.y = -1;
	}
}

void Viewer::Cuda_Verify(const char * str, cudaError_t & err) {
	if (err == cudaSuccess) return;
	throw CudaException(str, std::stoi(cudaGetErrorString(err)));
}

template<typename T>
void Viewer::call_kernel() {

	switch (mode)
	{
	case 0: {
		Mandel_setup<T> << <entire_block, xyblock >> > (dest_dev, (T)scale, center, res, max_iters);
		break;
	}
	case 1: {
		TComplex<T> c = make_complex((T)( - .618), (T)0);
		Julia_setup<T> << <entire_block, xyblock >> > (dest_dev, (T)scale, center, c, res, max_iters);
		break;
	}
	case 2: {
		TComplex<T> c = make_complex((T)( - .8), (T).156);
		Julia_setup<T> << <entire_block, xyblock >> > (dest_dev, (T)scale, center, c, res, max_iters);
		break;
	}
	case 3: {
		animate_value += .005;
		animate_value = animate_value > (6.283185) ? 0 : animate_value; // > 2pi
		//e^ix = cos(x) + i sin(x) by eulers formula
		TComplex<T> c = make_complex((T)(.7885 * cos(animate_value)), (T)(.7885 * sin(animate_value)));

		Julia_setup<T> << <entire_block, xyblock >> > (dest_dev, (T)scale, center, c, res, max_iters);
		break;
	}
	default:
		break;
	}
}

void Viewer::sync() {
	cudaError_t err = cudaDeviceSynchronize();
	Cuda_Verify("cudaDeviceSynchronize returned error", err);

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

	res.x = window.getSize().x;
	res.y = window.getSize().y;
	
	res.x = std::min(res.x, 3840); //clamp values
	res.y = std::min(res.y, 2160);

	//adjust so multiple of 16
	res.x -= res.x % thread_size;
	res.y -= res.y % thread_size;

	center.x = 0;
	center.y = 0;
	scale = 2;

	img_display.create(res.x, res.y); //bad, inneficient,
	sprite = sf::Sprite(img_display);

	sz_total = res.x * res.y * num_channels;

	aspect_ratio = res.x / (float)res.y;

	int x_count = 0, y_count = 0;

	if (res.x != 0) {
		x_count = (int)((res.x - 1) /thread_size) + 1;
	}
	if (res.y != 0) {
		y_count = (int)((res.y - 1) / thread_size) + 1;
	}

	entire_block = dim3(x_count,y_count,1);//resize window

	sf::FloatRect visibleArea(0, 0, res.x, res.y);
	window.setView(sf::View(visibleArea));

 

}

void Viewer::free_resources(uint8_t* dest_dev, sf::Uint8* tmp) {
	cudaFree(dest_dev);
	delete[] tmp;

}
