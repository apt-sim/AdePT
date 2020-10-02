// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file vec3.h
 * @brief represents a 3-vector and provides functionality to manipulate 3-vectors.
 * @author Davide Costanzo (d.costanzo@sheffield.ac.uk) and Mark Hodgkinson (d.costanzo@sheffield.ac.uk)
*/

#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
  public:
    /** @brief nominal constructor */
    vec3() {}

    /** @brief constructor which sets the 3 components of the vector.
     * e0, e1 and e2 are the components of a 3-vector
     */
    vec3(float e0, float e1, float e2) { m_e[0] = e0; m_e[1] = e1; m_e[2] = e2; }

    /** @brief returns the stored first component of the 3-vector */
    inline float x() const { return m_e[0]; }

    /** @brief returns the stored second component of the 3-vector */
    inline float y() const { return m_e[1]; }

    /** @brief returns the stored third component of the 3-vector */
    inline float z() const { return m_e[2]; }

    /** @brief returns reference to this vector */
    inline const vec3& operator+() const { return *this; }

    /** @brief returns reference to this vector, after adding an other vector to it.
     * v2 is the vector to be added to this vecor.
     * */
    inline vec3& operator+=(const vec3 &v2);

    /** @brief returns reference to this vector, after multiplying it by a value.
     * t is the value to multiple this vector by.
     * */
    inline vec3& operator*=(const float t);

    /** @brief returns the magnitude of the vector */
    inline float length() const { return sqrt(m_e[0]*m_e[0] + m_e[1]*m_e[1] + m_e[2]*m_e[2]); }

    /** @brief reduces this vector by the energy a particle loses
     * en is the energy lost.
     * Each component of the vector is reduced by en * 1/magnitude * component.
     */
    inline void energyLoss(float en);

    /** @brief scale each component of the vector by a value.
     * scale is the value to scale each component by
     */
    inline void scaleLength(float scale) {m_e[0]*=scale; m_e[1]*=scale; m_e[2]*=scale;  }

    float m_e[3]; ///< array to represent the 3-vector.
 };

/** @brief returns an output stream that prints the components of the vector
 * os is the ostream operator and t is the 3vec to print components of.
 */
inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
  os << t.m_e[0] << " " << t.m_e[1] << " " << t.m_e[2];
  return os;
}

inline void vec3::energyLoss(float en) {
  float k = 1 / sqrt(m_e[0]*m_e[0] + m_e[1]*m_e[1] + m_e[2]*m_e[2]);
  m_e[0]=m_e[0]-en*k*m_e[0];
  m_e[1]=m_e[1]-en*k*m_e[1];
  m_e[2]=m_e[2]-en*k*m_e[2];
}

/** @brief add two vectors together.
 * v1 and v2 are the vec3 to add.
 * returns the sum of the two vectors v1 and v2.
 */ 
inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.m_e[0] + v2.m_e[0], v1.m_e[1] + v2.m_e[1], v1.m_e[2] + v2.m_e[2]);
}

/** @brief multiply a vector by a value.
 * t is the value to multiply a 3-vector v by.
 * returns a new vector, which is the result of the operation.
 */ 
inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t*v.m_e[0], t*v.m_e[1], t*v.m_e[2]);
}

/** @brief multiply a vector by a value.
 * t is the value to multiply a 3-vector v by.
 * returns a new vector, which is the result of the operation.
 */ 
inline vec3 operator*(const vec3 &v, float t) {
  return vec3(t*v.m_e[0], t*v.m_e[1], t*v.m_e[2]);
}

inline vec3& vec3::operator+=(const vec3 &v){
  m_e[0]  += v.m_e[0];
  m_e[1]  += v.m_e[1];
  m_e[2]  += v.m_e[2];
  return *this;
}

inline vec3& vec3::operator*=(const float t) {
  m_e[0]  *= t;
  m_e[1]  *= t;
  m_e[2]  *= t;
  return *this;
}

#endif // VEC3_H

