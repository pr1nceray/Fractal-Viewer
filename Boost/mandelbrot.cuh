#include <iostream>
#include "CuComplex_alt.cuh"

__inline__ __device__ float3 get_color(float v_init) {
    float3 a_init = make_float3(.5, .5, .5);
    float3 b_init = make_float3(.5, .5, .5);
    float3 c_init = make_float3(1, 1, 1);
    float3 d_init = make_float3(.3, .2, .2);
    float tmp_x = a_init.x + (b_init.x * cos(6.2831 * (c_init.x * v_init + d_init.x)));
    float tmp_y = a_init.y + (b_init.y * cos(6.2831 * (c_init.y * v_init + d_init.y)));
    float tmp_z = a_init.z + (b_init.z * cos(6.2831 * (c_init.z * v_init + d_init.z)));
    return make_float3(tmp_x, tmp_y, tmp_z);
}




template <typename T> 
__inline__ __device__ var2<T> get_bounds(int true_x, int true_y, T scale, var2<int> res, var2<double> center) {
    T x_adjusted = ((((true_x / (float)res.x) - .5) * scale) * (float)res.x/res.y) + center.x; //adjust for resizability
    T y_adjusted = (-1.0 * (((true_y / (float)res.y) - .5) * scale)) + center.y;
    return var2<T>{ x_adjusted,y_adjusted };

}

__inline__ __device__ void write_array(uint8_t* dest, var2<int> idx , float3 rgb, int res_x) {

    dest[(idx.y * res_x * 4) + (idx.x * 4) + 0] = 255 * rgb.x;
    dest[(idx.y * res_x * 4) + (idx.x * 4) + 1] = 255 * rgb.y;
    dest[(idx.y * res_x * 4) + (idx.x * 4) + 2] = 255 * rgb.z;
    dest[(idx.y * res_x * 4) + (idx.x * 4) + 3] = 255;
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
__device__ uint8_t calculateJulia(var2<T> center, TComplex<T> c, int max_iters) {    
    TComplex<T> z_it = make_complex(center.x, center.y);

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
__global__ void Mandel_setup(uint8_t* dest, T scale, var2<double> center, var2<int> res, int max_iters)
{

    const float aspect_ratio = (float)res.x / res.y;

    //x,y coords in image ARRAY
    var2<int> index = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };

    //x,y coords in image in terms of plotting
    var2<T> coords = get_bounds(index.x, index.y, scale, res, center);

    uint8_t mandel_value = calculateMandel(coords, max_iters);

    float3 ret_color = get_color(mandel_value / 255.0);

    write_array(dest, index, ret_color, res.x);
}




template<typename T>
__global__ void Julia_setup(uint8_t* dest, T scale, var2<double> center, TComplex<T> complex, var2<int> res, int max_iters)
{
    T c_real = -0.618;
    T c_imagine = 0;
    TComplex<T> c = make_complex(c_real, c_imagine);

    const float aspect_ratio = (float)res.x / res.y;

    var2<int> index = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
   
    var2<T> coords = get_bounds(index.x, index.y, scale, res, center);

    uint8_t julia_value = calculateJulia(coords, complex, max_iters);
    float3 ret_color = get_color(julia_value / 255.0);

    write_array(dest, index, ret_color, res.x);
    
    }

