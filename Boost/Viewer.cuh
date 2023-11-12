
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics.hpp>

#include "mandelbrot.cuh" //include clause better
#include <iostream>


class Viewer {
	public:
		Viewer();
		void resize();
		void display();
		~Viewer();

	private:
		void Set_Defaults();
		void Check_Events();
		void Check_Keyboard();
		void free_resources(uint8_t* dest_dev, sf::Uint8* tmp);
		template<typename T>
		void call_kernel();

		void sync();
		void update_display();

		sf::Uint8* img_host = nullptr; // image on host
		uint8_t* dest_dev = nullptr; //image on device
		sf::Texture img_display;	//image to display
		sf::Sprite sprite;			//sprite
		sf::RenderWindow window;


		int res_x = -1;
		int res_y = -1;
		int num_channels;
		int sz_total;



		dim3 xyblock = dim3(1024, 1); //default (needs to be changed)
		dim3 entire_block = dim3(1, 512); //default (needs to be changed)


		int mode = 0;
		float center_x = 0;
		float center_y = 0;
		float scale = 2;

		int max_iters = 256;

};