#include "CuComplex_alt.cuh"

/*
* Operators for the cuComplex class.
*/

/*
* Operators between Complex object and int
*/
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

/*
* Operators between two Complex objects
*/
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
template<typename T>
__host__ __device__  __inline__ TComplex<T> operator-(const TComplex<T>& a, const TComplex<T>& b) {
    return { a.real - b.real, a.Imagine - b.Imagine };
}
