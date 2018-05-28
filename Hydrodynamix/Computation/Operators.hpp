/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Hydrodynamix
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <cuda_runtime.h>
#include <math.h>

namespace Computation {
    // Addition
    inline __host__ __device__
    float4 operator + (const float4& a, float b) {
        return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
    }

    inline __host__ __device__
    float4 operator + (const float4& a, const float4& b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    inline __host__ __device__
    float4& operator += (float4& a, float b) {
        a = a + b;
        return a;
    }

    inline __host__ __device__
    float4& operator += (float4& a, const float4& b) {
        a = a + b;
        return a;
    }

    // Subtraction
    inline __host__ __device__
    float4 operator - (const float4& a, float b) {
        return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
    }

    inline __host__ __device__
    float4 operator - (const float4& a, const float4& b) {
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    }

    inline __host__ __device__
    float4& operator -= (float4& a, float b) {
        a = a - b;
        return a;
    }

    inline __host__ __device__
    float4& operator -= (float4& a, const float4& b) {
        a = a - b;
        return a;
    }

    // Multiplication
    inline __host__ __device__
    float4 operator * (const float4& a, float b) {
        return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
    }

    inline __host__ __device__
    float4 operator * (float b, const float4& a) {
        return a * b;
    }

    inline __host__ __device__
    float4 operator * (const float4& a, const float4& b) {
        return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    }

    inline __host__ __device__
    float4& operator *= (float4& a, float b) {
        a = a * b;
        return a;
    }

    inline __host__ __device__
    float4& operator *= (float4& a, const float4& b) {
        a = a * b;
        return a;
    }

    // Division
    inline __host__ __device__
    float4 operator / (const float4& a, float b) {
        return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
    }

    inline __host__ __device__
    float4 operator / (const float4& a, const float4& b) {
        return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
    }

    inline __host__ __device__
    float4& operator /= (float4& a, float b) {
        a = a / b;
        return a;
    }

    inline __host__ __device__
    float4& operator /= (float4& a, const float4& b) {
        a = a / b;
        return a;
    }

    // Helper
    inline __host__ __device__
    float dot(const float4& a, const float4& b) {
        return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }

    inline __host__ __device__
    float dot3D(const float4& a, const float4& b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    inline __host__ __device__
    float dot2D(const float4& a, const float4& b) {
        return a.x*b.x + a.y*b.y;
    }

    inline __host__ __device__
    float4 cross(const float4& a, const float4& b) {
        return make_float4(
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x,
            0);
    }

    inline __host__ __device__
    float length(const float4& v) {
        return sqrtf(dot(v, v));
    }

    inline __host__ __device__
    float length3D(const float4& v) {
        return sqrtf(dot3D(v, v));
    }

    inline __host__ __device__
    float length2D(const float4& v) {
        return sqrtf(dot2D(v, v));
    }

    inline __host__ __device__
    float lensq(const float4& v) {
        return dot(v, v);
    }

    inline __host__ __device__
    float lensq3D(const float4& v) {
        return dot3D(v, v);
    }

    inline __host__ __device__
    float lensq2D(const float4& v) {
        return dot2D(v, v);
    }

    inline __host__ __device__
    float4 normalize(const float4& v) {
        float len = length(v);
        if (len < 0.00001f) {
            return v;
        }
        return v / len;
    }
}

#endif
