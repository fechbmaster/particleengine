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

#ifndef KERNEL_HELPER_CUH
#define KERNEL_HELPER_CUH

#include "CudaHelper.hpp"

#include <cuda_runtime.h>

#define USE_TEXTURE_MEMORY 1

#if USE_TEXTURE_MEMORY
#define GET_VALUE(s, t, i) tex1Dfetch(t##Tex, i)
#else
#define GET_VALUE(s, g, i) s.g[i]
#endif

namespace Computation {
    __host__ __device__
    int3 computeGridPosition(const float3& particlePos, const float3& worldSizeHalf, float cellSize);

    __host__ __device__
    unsigned int computeGridHash(const int3& gridPos, unsigned int gridSize);

    void fillFloat4Array(float4* deviceArray, unsigned int count, const float4& value);

#ifdef __CUDACC__
    template <class T, class... Args>
    void callCudaKernel(const T& func, int count, const Args&... args) {
        int blockSize, minGridSize, gridSize;
        CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, func, 0, count));

        gridSize = (count + blockSize - 1) / blockSize;
        func<<<gridSize, blockSize>>>(args...);
    }
#endif

#define COPY_TO_SYMBOL(symbol, src, count) \
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(symbol, src, count));

#define COPY_TO_SYMBOL_ASYNC(symbol, src, count) \
    CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(symbol, src, count));

#define BIND_TEXTURE(tex, devPtr, size) \
    CUDA_SAFE_CALL(cudaBindTexture(nullptr, tex, devPtr, size));

#define UNBIND_TEXTURE(tex) \
    CUDA_SAFE_CALL(cudaUnbindTexture(tex));
}

#endif
