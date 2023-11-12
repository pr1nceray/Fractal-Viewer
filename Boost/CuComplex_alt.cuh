#include <cuda_runtime.h>
#include <device_launch_parameters.h>

 template<typename T>  struct TComplex
{
    T real;
    T Imagine;
};

template<typename T>
__host__ __device__  TComplex<T> make_complex(T real, T imagine) {
    return { real,imagine };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> mult_complex(TComplex<T>& a, TComplex<T>& b) {
    return { a.real * b.real - b.Imagine * a.Imagine, a.Imagine * b.real + a.real * b.Imagine };
}

template<typename T>
__host__ __device__ __inline__ TComplex<T> add_complex(TComplex<T>& a, TComplex<T>& b) {
    return { a.real + b.real, a.Imagine + b.Imagine };
}

template<typename T>
__host__ __device__ __inline__ T abs_nonsqrt_complex(TComplex<T>& a) {
    return { (a.real * a.real + a.Imagine * a.Imagine) };
}

template<typename T>
__host__ __device__ __inline__ T abs_complex(TComplex<T>& a) {
    return { sqrt(a.real * a.real + a.Imagine * a.Imagine) };
}

template<typename T>  struct var2
{
    T x;
    T y;
};


