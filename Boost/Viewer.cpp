
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics.hpp>

#include "mandelbrot.cuh" //include clause better



class Viewer {
	public : 
	Viewer() {
		cudaError_t err;
		
		img_host = new sf::Uint8[sz_total];

		err = cudaMalloc((void**)&dest_dev, sz_total * sizeof(sf::Uint8));

		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc launch failed: %s\n", cudaGetErrorString(err));
			free_resources(dest_dev, img_host);		//MAYBE CHANGE?
		}

		img_display.create(res_x, res_y);
		sprite = sf::Sprite(img_display);

		window.create(sf::VideoMode(1025, 512), "Fractal Viewer2");
		window.setFramerateLimit(60); // maybe remove?

	}


	void resize() {
	//update info	
	}

	private : 
		

	sf::Uint8* img_host = nullptr; // image on host
	uint8_t* dest_dev = nullptr; //image on device
	sf::Texture img_display;	//image to display
	sf::Sprite sprite;			//sprite
	sf::RenderWindow window;


	int res_x = 1024;
	int res_y = 512;
	int num_channels = 4;
	int sz_total = res_x * res_y * num_channels;



	dim3 xyblock = dim3(1024, 1); //default (needs to be changed)
	dim3 entire_block = dim3(1, 512); //default (needs to be changed)


	int mode = 0;

	

	void free_resources(uint8_t* dest_dev, sf::Uint8* tmp) {
		cudaFree(dest_dev);
		delete[] tmp;

	}

};