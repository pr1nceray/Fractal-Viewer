#include <iostream>
#include "CuComplex_alt.cuh"

__inline__ __device__ float3 get_color(float v_init) {
    float3 a_init = make_float3(.8, .5, .4);
    float3 b_init = make_float3(.2, .4, .2);
    float3 c_init = make_float3(2, 1, 1);
    float3 d_init = make_float3(0, .25, .25);
    float tmp_x = a_init.x + (b_init.x * cos(6.2831 * (c_init.x * v_init + d_init.x)));
    float tmp_y = a_init.y + (b_init.y * cos(6.2831 * (c_init.y * v_init + d_init.y)));
    float tmp_z = a_init.z + (b_init.z * cos(6.2831 * (c_init.z * v_init + d_init.z)));
    return make_float3(tmp_x, tmp_y, tmp_z);
}



template <typename T> 
__inline__ __device__ __host__ var2<T> get_bounds(int true_x, int true_y, T scale, T center_x, T center_y) {
    T x_adjusted = ((((true_x / 1024.0) - .5) * scale) * 2) + center_x; //we multiply by 2 for aspect ratio
    T y_adjusted = (-1.0 * (((true_y / 512.0) - .5) * scale)) + center_y;
    return var2<T>{ x_adjusted,y_adjusted };

}

template<typename T>
__device__ uint8_t calculateMandel(var2<T> center, int max_iters) {
    TComplex<T> z_it = make_complex((T)0, (T)0);
    TComplex<T> c = make_complex(center.x, center.y);
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
__device__ uint8_t calculateJulia(var2<T> center, T complex_val,T max_iters) {    
    TComplex<T> z_it = make_complex(center.x, center.y);
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

    int true_x = /*blockIdx.x * blockDim.x + */threadIdx.x;
    int true_y = /*blockIdx.y * blockDim.y + */blockIdx.y;

    T cur_x = ((true_x / 1024.0) - .5) * scale;// *width; //-width/2 to width/2
    T cur_y = -1.0 * (((true_y / 512.0) - .5) * scale); //.height/2 to -height/2


    cur_x *= 2; //2 for aspect ratio

    cur_x += center_x;
    cur_y += center_y;

    //uint8_t mandel_value = calculateJulia<T>(get_bounds(true_x, true_y, scale, center_x, center_y), -1.3, max_iters);

    uint8_t mandel_value = calculateMandel<T>(get_bounds(true_x,true_y,scale,center_x,center_y), max_iters);

    float3 ret_color = get_color(mandel_value / 255.0);
    dest[(true_y * 1024 * 4) + (true_x * 4) + 0] = 255 * ret_color.x;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 1] = 255 * ret_color.y;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 2] = 255 * ret_color.z;
    dest[(true_y * 1024 * 4) + (true_x * 4) + 3] = 255;
}

