//for p(z) = z^5 * sin(z) -1
//p'(z) = 5z^4 * sin(z) + cos(z) *z^5
#include "CuComplex_operations.cuh"
#include <iostream>

__inline__ __device__ float3 get_color(float v_init) {
    float3 a_init = make_float3(.5, .5, .5);
    float3 b_init = make_float3(.5, .5, .5);
    float3 c_init = make_float3(1, 1, 1);
    float3 d_init = make_float3(0, .1, .2);
    float tmp_x = a_init.x + (b_init.x * cos(6.2831 * (c_init.x * v_init + d_init.x)));
    float tmp_y = a_init.y + (b_init.y * cos(6.2831 * (c_init.y * v_init + d_init.y)));
    float tmp_z = a_init.z + (b_init.z * cos(6.2831 * (c_init.z * v_init + d_init.z)));
    return make_float3(tmp_x, tmp_y, tmp_z);
}


template <typename T>
__inline__ __device__ var2<T> get_bounds(int true_x, int true_y, T scale, var2<int> res, var2<double> center) {
    T x_adjusted = ((((true_x / (float)res.x) - .5) * scale) * (float)res.x / res.y) + center.x; //adjust for resizability
    T y_adjusted = (-1.0 * (((true_y / (float)res.y) - .5) * scale)) + center.y;
    return var2<T>{ x_adjusted, y_adjusted };

}

__inline__ __device__ void write_array(uint8_t* dest, var2<int> idx, float3 rgb, int res_x) {

    dest[(idx.y * res_x * 4) + (idx.x * 4) + 0] = 255 * rgb.x;
    dest[(idx.y * res_x * 4) + (idx.x * 4) + 1] = 255 * rgb.y;
    dest[(idx.y * res_x * 4) + (idx.x * 4) + 2] = 255 * rgb.z;
    dest[(idx.y * res_x * 4) + (idx.x * 4) + 3] = 255;
}


//for p(z) = z^3 -1
//p'(z) = 3z^2
template<typename T>
__device__ uint8_t calculateNewtons_fast(var2<T> center, int max_iters) {
    const float epsilon = .005;
    TComplex<float> z_it = make_complex((float)center.x, (float)center.y);

    TComplex<float> p_prime;
    TComplex<float> p;
    const TComplex<float> temp_t = { 1,1 };
    for (int i = 0; i < max_iters; ++i) {
        p_prime = pow_complex_fast(z_it, 2);
        p_prime *= 3;

        p = pow_complex_fast(z_it, 3);
        p.real -= 1;

        z_it = sub_complex(z_it, (p / p_prime));
    }
    if ((abs(z_it.real - 1) < epsilon) && abs(z_it.Imagine) < epsilon) {
        return 0;
    }
    //if it goes to (-0.5 + 0.86602540378444i)
    else if (abs(z_it.Imagine - 0.86602540378444) < epsilon) {
        return 1;
    }
    return 2;
}

template<typename T>
__device__ uint8_t calculateNewtons(var2<T> center, int max_iters) {
    const float epsilon = .005;
    TComplex<T> z_it = make_complex(center.x, center.y);

    TComplex<T> p_prime;
    TComplex<T> p;
    const TComplex<float> temp_t = { 1,1 };
    for (int i = 0; i < max_iters; ++i) {
        p_prime = pow_complex(z_it, 2);
        p_prime *= 3;

        p = pow_complex(z_it, 3);
        p.real -= 1;

        z_it = sub_complex(z_it, (p / p_prime));
    }
    if ((abs(z_it.real - 1) < epsilon) && abs(z_it.Imagine) < epsilon) {
        return 0;
    }
    //if it goes to (-0.5 + 0.86602540378444i)
    else if (abs(z_it.Imagine - 0.86602540378444) < epsilon) {
        return 1;
    }
    else {
        return 2;
    }
}

//doesnt really work cuz of accumilated error when approximating causes it to shit the bed
template<typename T>
__device__ uint8_t calculateNewtons2_fast(var2<T> center, int max_iters) {
    const float epsilon = .0005;
    TComplex<float> z_it = make_complex((float)center.x, (float)center.y);

    TComplex<float> p_prime;
    TComplex<float> p;

    TComplex <float> p_sind;
    TComplex <float> p_cosd;

    TComplex <float> p_mult;

    const TComplex<float> temp_t = { 1,0 };
    for (int i = 0; i < max_iters; ++i) {

        p_sind = sin_complex_fast(z_it);
        p_cosd = cos_complex_fast(z_it);

        p = pow_complex_fast(z_it, 5);

        p_mult = p * p_sind; //p_mult = p^5 * sin(p)
        p_mult = sub_complex(p_mult, temp_t);//p_mult = p^5 * sin(p) -1

        p_prime = pow_complex_fast(z_it, 4);
        p_prime = (5 * p_prime * p_sind + p * p_cosd);

        z_it = sub_complex(z_it, (p_mult / p_prime));

        if (abs(abs(z_it.real) - 1.031) < epsilon) { //pos or negative, so we take abs ahead of time.
            return i;
        }
        if (abs(abs(z_it.real) - 3.138) < epsilon) {
            return i;
        }
        if (abs(abs(z_it.real) - 6.283) < epsilon) {
            return i;
        }
        if (abs(abs(z_it.real) - 9.424) < epsilon) {
            return i;
        }
    }

    return 0;
}

template<typename T>
__device__ uint8_t calculateNewtons2(var2<T> center, int max_iters) {
    const float epsilon = .000005;
    TComplex<T> z_it = make_complex(center.x, center.y);

    TComplex<T> p_prime;
    TComplex<T> p;

    TComplex <T> p_sind;
    TComplex <T> p_cosd;

    TComplex <T> p_mult;

    const TComplex<T> temp_t = { 1,0 };
    for (int i = 0; i < max_iters; ++i) {

        p_sind = sin_complex(z_it);
        p_cosd = cos_complex(z_it);

        p = pow_complex(z_it, 5);

        p_mult = p * p_sind; //p_mult = p^5 * sin(p)
        p_mult = sub_complex(p_mult, temp_t);//p_mult = p^5 * sin(p) -1

        p_prime = pow_complex(z_it, 4);
        p_prime = (5 * p_prime * p_sind + p * p_cosd);

        z_it = sub_complex(z_it, (p_mult / p_prime));

        if (abs(pow(z_it.real, 5) * sin(z_it.real) - 1) < epsilon) {
            return i;
        }
    }

    return 0;
}

template<typename T>
__global__ void Newton_setup(uint8_t* dest, T scale, var2<double> center, var2<int> res, int max_iters,int newton_selection)
{
    const float aspect_ratio = (float)res.x / res.y;

    var2<int> index = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };

    //x,y coords in image in terms of plotting
    var2<T> coords = get_bounds(index.x, index.y, scale, res, center);
    
    uint8_t newtons;
    float3 ret_color; 
    switch(newton_selection){
        case 0: {
            newtons = calculateNewtons_fast(coords, max_iters);
            ret_color = make_float3(newtons ==0 ? 1 : 0, newtons == 1 ? 1 : 0, newtons == 2 ? 1 : 0);
            break;
        }
        case 1: {
            newtons = calculateNewtons2(coords, max_iters);
            ret_color = get_color(newtons / 128.0);
            break;
        }
    }

    write_array(dest, index, ret_color, res.x);
}


