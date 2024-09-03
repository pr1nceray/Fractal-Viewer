
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics.hpp>

#include "stb_image_write.h"
#include "mandelbrot.cuh" //include clause better
#include <iostream>

#define sz_max 3840 * 2160 * 4

class Viewer {
	public:
		Viewer();
		void resize();
		void display();
		~Viewer();

	private:

		void Init_Menu();
		void Set_Defaults();
		void Check_Events();
		void Check_Keyboard();
		void Check_Mouse();

		void free_resources(uint8_t* dest_dev, sf::Uint8* tmp);
		template<typename T>
		void call_kernel();

		void Cuda_Verify(const char * str, cudaError_t& e);

		void sync();
		void update_display();

		sf::Uint8* img_host = nullptr; // image on host
		uint8_t* dest_dev = nullptr; //image on device
		sf::Texture img_display;	//image to display
		sf::Sprite sprite;			//sprite
		sf::RenderWindow window;
		sf::Vector2i last_mouse;


		var2<int> res;

		float aspect_ratio = 2;
		int num_channels;
		int sz_total;


		const int thread_size = 16;

		dim3 xyblock = dim3(thread_size,thread_size); //create 8 warps, 256 threads 
		dim3 entire_block = dim3(64, 32); //ampere (my gpu has 64 warps per sm, 2048 threads. cleanly divisible by xyblock! each sm block takes 8 xyblocks! this takes 64 * 4 sm cores (lol sadge) 


		int mode = 1;

		var2<double> center;
		double scale = 2;
		bool precise = true;
		
		
		int max_iters = 256;
		int newton_iters = 128;
		
		float animate_value = 0;
		float coolDownPress = 0.25f;
		bool pausedAnimation = false;
		bool invertAnimation = false;

		class CudaException : public std::exception {
			public : 
				std::string msg;
				int code;
				CudaException(const std::string & message_in, int code_in) : msg(message_in), code(code_in) {
					
				}

				virtual std::string what() {
					return msg;
				}
		};
};

