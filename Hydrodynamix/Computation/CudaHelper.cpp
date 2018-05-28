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

#include "StdAfx.hpp"
#include "CudaHelper.hpp"

#include <cstdint>
#include <cuda_gl_interop.h>
#include <QHash>

namespace Computation {
    static QHash<void*, size_t> memoryBlockTable;

    void checkCudaError(cudaError err, const char* file, int line) {
        if (err != cudaSuccess) {
            qDebug() << "CUDA error in file" << file << "@ line"
                << line << ":" << cudaGetErrorString(err);
        }
    }

    void checkForDeviceMemoryLeaks() {
        if (!memoryBlockTable.empty()) {
            qDebug() << "====================================";
            qDebug() << "WARNING: Detected CUDA memory leaks!";
            qDebug() << "------------------------------------";
        }

        auto itr = memoryBlockTable.begin();
        auto end = memoryBlockTable.end();

        for (; itr != end; ++itr) {
            qDebug() << "Block @" << itr.key() << "size:" << itr.value();
        }

        if (!memoryBlockTable.empty()) {
            qDebug() << "====================================";
        }
    }

    void* mallocDevice(size_t size) {
        void* ptr = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&ptr, size));
        if (ptr) {
            fillValue(ptr, 0, size);
            memoryBlockTable[ptr] = size;
        }
        return ptr;
    }

    void* reallocDevice(void* ptr, size_t size) {
        size_t currentSize = memoryBlockTable.value(ptr);
        if (currentSize >= size) {
            return ptr;
        }

        size_t newSize = size + size / 2;
        void* buffer = mallocDevice(newSize);

        if (!buffer) {
            buffer = mallocDevice(size);
        }

        if (buffer && ptr) {
            moveDeviceToDevice(buffer, ptr, currentSize);
            freeDevice(ptr);
        }
        return buffer;
    }

    void freeDevice(void* ptr) {
        if (ptr) {
            memoryBlockTable.remove(ptr);
            CUDA_SAFE_CALL(cudaFree(ptr));
        }
    }

    void moveHostToDevice(void* target, const void* source, size_t count) {
        CUDA_SAFE_CALL(cudaMemcpy(target, source, count, cudaMemcpyHostToDevice));
    }

    void moveDeviceToHost(void* target, const void* source, size_t count) {
        CUDA_SAFE_CALL(cudaMemcpy(target, source, count, cudaMemcpyDeviceToHost));
    }

    void moveDeviceToDevice(void* target, const void* source, size_t count) {
        CUDA_SAFE_CALL(cudaMemcpy(target, source, count, cudaMemcpyDeviceToDevice));
    }

    void fillValue(void* ptr, int value, size_t count) {
        CUDA_SAFE_CALL(cudaMemset(ptr, value, count));
    }

    void deviceSync() {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    void registerGraphicsResource(cudaGraphicsResource** graphicsResource, unsigned int bufferID) {
        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(graphicsResource, bufferID, cudaGraphicsMapFlagsNone));
    }

    void unregisterGraphicsResource(cudaGraphicsResource* graphicsResource) {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(graphicsResource));
    }

    void* mapGraphicsResource(cudaGraphicsResource** graphicsResource) {
        void* ptr = nullptr;
        size_t numBytes = 0;
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, graphicsResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &numBytes, *graphicsResource));
        return ptr;
    }

    void unmapGraphicsResource(cudaGraphicsResource* graphicsResource) {
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &graphicsResource, 0));
    }

    void startCudaTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&stop));
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    }

    float stopCudaTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        float cudaTime = 0;
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&cudaTime, start, stop));
        return cudaTime;
    }
}
