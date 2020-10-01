#include <math.h>
#include <stdlib.h>
#include <iostream>
#ifdef __CUDACC__
#include "cuda.h"
#endif

#ifndef VEC3_H
#define VEC3_H


class vec3 {
  public:
        vec3() {}

        vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

        inline float x() const { return e[0]; }

        inline float y() const { return e[1]; }

        inline float z() const { return e[2]; }

        inline const vec3& operator+() const { return *this; }

        inline vec3& operator+=(const vec3 &v2);

        inline vec3& operator*=(const float t);

        inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }

        inline void energyLoss(float en);

        inline void scaleLength(float scale) {e[0]*=scale; e[1]*=scale; e[2]*=scale;  }

        float e[3];
 };


inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

inline void vec3::energyLoss(float en) {
    float k = 1 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0]=e[0]-en*k*e[0];
    e[1]=e[1]-en*k*e[1];
    e[2]=e[2]-en*k*e[2];
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3& vec3::operator+=(const vec3 &v){
     e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

inline vec3& vec3::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

#endif // VEC3_H

