#include "CuComplex_alt.cuh"


//operations for complex with an int
template<typename T>
__host__ __device__  __inline__ TComplex<T> operator/(const TComplex<T>& b, const int a) {
    return { b.real / x, b.Imagine / a };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator*(const TComplex<T>& b, const int a) {
    return { b.real * a, b.Imagine * a };
}


template<typename T>
__host__ __device__  __inline__ TComplex<T> operator*(const int a, const TComplex<T>& b) {
    return { b.real * a, b.Imagine * a };
}

template<typename T>
__host__ __device__  __inline__ TComplex<T>& operator*=(TComplex<T>& a, const int b) {
    a.real *= b; a.Imagine *= b; return a;
}

//operations between 2 complex numbers
template<typename T>
__host__ __device__  __inline__ TComplex<T> operator/(const TComplex<T>& a, const TComplex<T>& b) {
    return div_complex(a, b);
}

template<typename T>
__host__ __device__  __inline__ TComplex<T> operator*(const TComplex<T>& a, const TComplex<T>& b) {
    return mult_complex(a, b);
}


template<typename T>
__host__ __device__  __inline__ TComplex<T> operator+(const TComplex<T>& a, const TComplex<T>& b) {
    return { a.real + b.real, a.Imagine + b.Imagine };
}
