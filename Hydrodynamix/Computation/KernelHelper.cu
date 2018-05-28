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

#include "KernelHelper.cuh"

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Computation {
    __host__ __device__
    int3 computeGridPosition(const float3& particlePos, const float3& worldSizeHalf, float cellSize) {
        int3 gridPos;
        gridPos.x = floor((particlePos.x - (-worldSizeHalf.x)) / cellSize);
        gridPos.y = floor((particlePos.y - (-worldSizeHalf.y)) / cellSize);
        gridPos.z = floor((particlePos.z - (-worldSizeHalf.z)) / cellSize);
        return gridPos;
    }

    __host__ __device__
    unsigned int computeGridHash(const int3& gridPos, unsigned int gridSize) {
        unsigned int x = gridPos.x & (gridSize - 1);
        unsigned int y = gridPos.y & (gridSize - 1);
        unsigned int z = gridPos.z & (gridSize - 1);

        return z * gridSize * gridSize + y * gridSize + x;
    }

    __global__ void fillFloat4Kernel(float4* deviceArray, unsigned int count, float4 value) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < count) {
            deviceArray[tid] = value;
        }
    }

    void fillFloat4Array(float4* deviceArray, unsigned int count, const float4& value) {
        callCudaKernel(fillFloat4Kernel, count, deviceArray, count, value);
    }
}
