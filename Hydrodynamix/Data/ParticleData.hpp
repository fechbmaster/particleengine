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

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "Computation/CudaPointer.hpp"

namespace Data {
    struct ParticleData {
        Computation::CudaPointer<float4> position;
        Computation::CudaPointer<float4> velocity;
        Computation::CudaPointer<float4> force;
        Computation::CudaPointer<float4> surfaceNormal;
        Computation::CudaPointer<float> density;
        Computation::CudaPointer<float> colorField;

        Computation::CudaPointer<float4> sortedPosition;
        Computation::CudaPointer<float4> sortedVelocity;
        Computation::CudaPointer<float> sortedDensity;
    };
}

#endif
