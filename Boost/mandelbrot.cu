/*
#include "mandelbrot.cuh"

__device__ uint8_t calculateMandel(double cur_x, double cur_y,int max_iters) {
    cuDoubleComplex z_it = make_cuDoubleComplex(0, 0);
    cuDoubleComplex c = make_cuDoubleComplex(cur_x, cur_y);
    int count = 1;

    for (int i = 0; i < max_iters; ++i) {
        if (cuCabs(z_it) > 2) {
            return count;
        }
        z_it = cuCadd(cuCmul(z_it, z_it), c);
        count++;
    }
    return 255;
}


__device__ uint8_t calculateJulia(double cur_x, double cur_y, int max_iters) {
    cuDoubleComplex c = make_cuDoubleComplex(0.27334, .73);
    cuDoubleComplex z_it = make_cuDoubleComplex(0, 0);
    int count = 1;

    for (int i = 0; i < max_iters; ++i) {
        if (cuCabs(z_it) > 2) {
            return count;
        }
        z_it = cuCadd(cuCmul(z_it, z_it), c);
        count++;
    }
    return 255;
}

__global__ void Determine_ends(uint8_t* dest, double scale, double center_x, double center_y,int max_iters)
{

    int true_x = threadIdx.x;
    int true_y = blockIdx.y;

    double cur_x = ((true_x / 1024.0) - .5) * scale;// *width; //-width/2 to width/2
    double cur_y = -1.0 * (((true_y / 512.0) - .5) * scale); //.height/2 to -height/2

    cur_x *= 2; //2 for aspect ratio

    cur_x += center_x;
    cur_y += center_y;

    uint8_t mandel_value = calculateJulia(cur_x, cur_y, max_iters);

    dest[(true_y * 1024*4) + (true_x * 4) + 0] = mandel_value;
    dest[(true_y * 1024*4) + (true_x * 4) + 1] = mandel_value;
    dest[(true_y * 1024*4) + (true_x * 4) + 2] = mandel_value;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 3] = 255;
}
*/