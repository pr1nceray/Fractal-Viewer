
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics.hpp>

#include <stdio.h>

#include "Viewer.cuh"

using namespace std;


int main(int argc, char** argv) {
    Viewer main_view;
    main_view.display();
}
