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
__host__ __device__ __inline__ TComplex<T> add_complex(const TComplex<T>& a, const TComplex<T>& b) {
    return { a.real + b.real, a.Imagine + b.Imagine };
}

template<typename T>
__host__ __device__ __inline__ TComplex<T> sub_complex(const TComplex<T>& a, const TComplex<T>& b) {
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
__host__ __device__ __inline__ float abs_fast_complex(TComplex<T>& a) {
    return { __fsqrt_rd(a.real * a.real + a.Imagine * a.Imagine) };
}

template<typename T>
 __device__ __inline__ TComplex<float> pow_complex_fast(TComplex<T> &a, int power) {
    float r = abs_fast_complex(a);//radius
    float theta  = atan2f(a.Imagine,a.real); //faster tahn tan
    float powed = __powf(r, power);
    return { powed * __cosf(power * theta), powed * __sinf(power * theta)};
}

 template<typename T>
 __device__ __inline__ TComplex<T> pow_complex(TComplex<T>& a, int power) {
     T r = abs_complex(a);//radius
     float theta = atan2f((float)a.Imagine, (float)a.real); //faster tahn tan
     T powed = pow(r, power);
     return { powed * cos(power * theta), powed * sin(power * theta) };
 }


 template<typename T>
 __device__ __inline__ TComplex<T> sin_complex(const TComplex<T>& a) {
     return { sin(a.real) * cosh(a.Imagine), cos(a.real) * sinh(a.Imagine) };
 }

 template<typename T>
 __device__ __inline__ TComplex<float> sin_complex_fast(const TComplex<T>& a) {
     return {__sinf(a.real) * cosh(a.Imagine), __cosf(a.real) * sinh(a.Imagine) };
 }

 template<typename T>
 __device__ __inline__ TComplex<T> cos_complex(const TComplex<T>& a) {
     return { cos(a.real) * cosh(a.Imagine), -1 * sin(a.real) * sinh(a.Imagine) };
 }

 template<typename T>
 __device__ __inline__ TComplex<float> cos_complex_fast(const TComplex<T>& a) {
     return { __cosf(a.real) * cosh(a.Imagine), -1 * __sinf(a.real) * sinh(a.Imagine) };
 }

template<typename T>  struct var2
{
    T x;
    T y;
};

__host__ __device__ __inline__ float3 operator*(const int &f, const float3& arr) {
    return make_float3(arr.x * f, arr.y * f, arr.z * f);
}
