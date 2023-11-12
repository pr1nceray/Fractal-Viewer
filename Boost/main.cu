
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics.hpp>

#include <stdio.h>

#include "Viewer.cuh"

/*
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/
using namespace std;







int main(int argc, char** argv) {
    Viewer main_view;
    main_view.display();
}


/*
int main(int argc,char ** argv) {
    
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        return -1;
    }

    int res_x = 1024;
    int res_y = 512;
    int num_channels = 4;
    int sz_total = res_x * res_y * num_channels;

    sf::Uint8* img = new sf::Uint8[sz_total];
    uint8_t* dest_dev = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&dest_dev, sz_total * sizeof(sf::Uint8));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc launch failed: %s\n", cudaGetErrorString(err));
        free_resources(dest_dev,img);
        return -1;
    }

    dim3 xyblock(1024, 1);
    dim3 entire_block(1, 512);

    float center_x = 0;
    float center_y = 0;
    float scale = 2;

    int max_iters = 256;

    int mode = 0;

    sf::Texture img_display;
    img_display.create(res_x, res_y);
    sf::Sprite sprite(img_display);


    
    sf::RenderWindow window;
    window.create(sf::VideoMode(1025, 512), "Fractal Viewerw2");
    window.setFramerateLimit(60);
    
    sf::Vector2i last_mouse(-1, -1);

    while (window.isOpen()) {
        switch (mode) {
            case 0: {
                Determine_ends<double> << <entire_block, xyblock >> > (dest_dev, scale, center_x, center_y, max_iters); //max time before we call again to stop it from blocking
            }
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(err));
            free_resources(dest_dev,img);
            return -1;
        }

        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) {
                window.close();
            }
            if (e.type == sf::Event::MouseWheelMoved) {
                double moved =  e.mouseWheel.delta;
                //cout << (((e.mouseWheel.x - 1024) / 1024.0) -  .5) * scale << " " << -1 * (((e.mouseWheel.y - 512) / 512.0) + .5) * scale * 2 << "\n";
                if (moved > 0) {
                    scale = scale/sqrt(abs(moved)+.025);
                }
                else {
                    scale = scale * sqrt(abs(moved)+.025);
                    scale = min(scale, (float)4.0);
                }
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
            center_y += scale/64.0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
            center_y -= scale/64.0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
            center_x -= scale/64.0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            center_x += scale/64.0;
        }


        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            auto mouse_pos = sf::Mouse::getPosition(window);

            if (last_mouse.x != -1) {
                center_x -= ((mouse_pos.x-last_mouse.x)/1024.0) * scale * 1.75; //flipped for x
                center_y += ((mouse_pos.y - last_mouse.y) /512.0) * scale * 1.25;
            }
            last_mouse = mouse_pos;
        }
        else {
            last_mouse.x = -1; last_mouse.y = -1;
        }

        if (abs(center_x) > 2) {
            center_x = 2 * (center_x < 0 ? -1 : 1);
        }
        if (abs(center_y) > 1) {
            center_y = 1 * (center_y < 0 ? -1 : 1);
        }
       

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
            free_resources(dest_dev,img);
            return -1;
        }
        cudaMemcpy(img, dest_dev, sz_total * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
        
        img_display.update(img);
        sprite.setTexture(img_display);
        
        window.clear();
        window.draw(sprite);
        window.display();
       
    }
   
    
    free_resources(dest_dev,img);
    return;

}

*/