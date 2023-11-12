#include <iostream>
#include "CuComplex_alt.cuh"


template<typename T>
__device__ uint8_t calculateMandel(T cur_x, T cur_y, int max_iters) {
    TComplex<T> z_it = make_complex((T)0, (T)0);
    TComplex<T> c = make_complex(cur_x, cur_y);
    int count = 1;

    for (int i = 0; i < max_iters; ++i) {
        if (abs_nonsqrt_complex(z_it) > 4.0) {
            return count;
        }
        z_it = add_complex(mult_complex(z_it, z_it), c);
        count++;
    }
    return 255;
}



template<typename T>
__device__ uint8_t calculateJulia(T cur_x, T cur_y, T complex_val,T max_iters) {    
    TComplex<T> z_it = make_complex(cur_x, cur_y);
    T c_real = -0.618;
    T c_imagine = 0;
    TComplex<T> c = make_complex(c_real, c_imagine);

    int count = 1;

    for (int i = 0; i < max_iters; ++i) {
        if ( abs_nonsqrt_complex(z_it) > 4.0) {
            return count;
        }
        z_it = add_complex(mult_complex(z_it, z_it), c);
        count++;
    }
    return 255;
}

template<typename T>
__global__ void Determine_ends(uint8_t* dest, T scale, T center_x, T center_y, int max_iters)
{

    int true_x = threadIdx.x;
    int true_y = blockIdx.y;

    T cur_x = ((true_x / 1024.0) - .5) * scale;// *width; //-width/2 to width/2
    T cur_y = -1.0 * (((true_y / 512.0) - .5) * scale); //.height/2 to -height/2


    cur_x *= 2; //2 for aspect ratio

    cur_x += center_x;
    cur_y += center_y;

    //uint8_t mandel_value = calculateJulia<T>(cur_x, cur_y, -1.3, max_iters);

    uint8_t mandel_value = calculateMandel<T>(cur_x, cur_y, max_iters);
    dest[(true_y * 1024 * 4) + (true_x * 4) + 0] = mandel_value;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 1] = mandel_value;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 2] = mandel_value;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 3] = 255;
}

