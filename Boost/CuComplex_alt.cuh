#include <cuda_runtime.h>
#include <device_launch_parameters.h>

 template<typename T>  struct TComplex
{
    T real;
    T Imagine;
};

template<typename T>
__host__ __device__  TComplex<T> make_complex(const T &real, const T &imagine) {
    return { real,imagine };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> mult_complex(const TComplex<T>& a, const TComplex<T>& b) {
    return { a.real * b.real - b.Imagine * a.Imagine, a.Imagine * b.real + a.real * b.Imagine };
}


template<typename T>
__host__ __device__  __inline__ TComplex<T> div_complex(const TComplex<T>& a, const TComplex<T>& b) {
    T div = b.real * b.real + b.Imagine * b.Imagine;
    return { (a.real * b.real + b.Imagine * a.Imagine)/(div), (a.Imagine * b.real - a.real * b.Imagine)/div};
}

template<typename T>
__host__ __device__ __inline__ TComplex<T> add_complex(TComplex<T>& a, TComplex<T>& b) {
    return { a.real + b.real, a.Imagine + b.Imagine };
}

template<typename T>
__host__ __device__ __inline__ TComplex<T> sub_complex(TComplex<T>& a, TComplex<T>& b) {
    return { a.real - b.real, a.Imagine - b.Imagine };
}


template<typename T>
__host__ __device__ __inline__ T abs_nonsqrt_complex(TComplex<T>& a) {
    return { (a.real * a.real + a.Imagine * a.Imagine) };
}

template<typename T>
__host__ __device__ __inline__ T abs_complex(TComplex<T>& a) {
    return { sqrt(a.real * a.real + a.Imagine * a.Imagine) };
}

template<typename T>
 __device__ __inline__ TComplex<T> pow_complex(TComplex<T> &a, int power) {
    float r = abs_complex(a);//radius
    T theta = 0;
    if (a.real != 0) {
        theta = atan(a.Imagine / a.real);
    }
    float powed = __powf(r, power);
    return { powed * cos(power * theta), powed * sin(power * theta)};
}

template<typename T>  struct var2
{
    T x;
    T y;
};

__host__ __device__ __inline__ float3 operator*(const int &f, const float3& arr) {
    return make_float3(arr.x * f, arr.y * f, arr.z * f);
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator/(const TComplex<T>& b , const int a) {
    return { b.real/x, b.Imagine /a };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator*(const int a, const TComplex<T>& b) {
    return { b.real * x, b.Imagine* a };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator*(const TComplex<T>& b, const int a) {
    return { b.real * a, b.Imagine * a };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator/(const TComplex<T>& a, const TComplex<T>& b) {
    return div_complex(a, b);
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator*(const TComplex<T>& a, const TComplex<T>& b) {
    return mult_complex(a, b);
}
