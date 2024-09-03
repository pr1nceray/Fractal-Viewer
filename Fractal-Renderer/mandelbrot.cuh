#include "Newtons.cuh"

template<typename T>
__device__ uint8_t calculateMandel(var2<T> center, int max_iters) {
    TComplex<T> z_it = make_complex((T)0, (T)0);
    TComplex<T> c = make_complex(center.x, center.y);

    for (int i = 0; i < max_iters; ++i) {
        if (abs_nonsqrt_complex(z_it) > 4.0) {
            return i;
        }
        z_it = add_complex(z_it * z_it, c);
    }
    return 255;
}


template<typename T>
__device__ uint8_t calculateJulia(var2<T> center, TComplex<T> c, int max_iters) {    
    TComplex<T> z_it = make_complex(center.x, center.y);

    for (int i = 0; i < max_iters; ++i) {
        if ( abs_nonsqrt_complex(z_it) > 4.0) {
            return i;
        }
        z_it = add_complex(mult_complex(z_it, z_it), c);
    }
    return 255;
}


template<typename T>
__global__ void Mandel_setup(uint8_t* dest, T scale, var2<double> center, var2<int> res, int max_iters)
{


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

    var2<int> index = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
   
    var2<T> coords = get_bounds(index.x, index.y, scale, res, center);

    uint8_t julia_value = calculateJulia(coords, complex, max_iters);
    float3 ret_color = get_color(julia_value / 255.0);

    write_array(dest, index, ret_color, res.x);
    
    }
