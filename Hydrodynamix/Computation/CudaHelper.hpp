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

#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include <new>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(err) \
    Computation::checkCudaError(err, __FILE__, __LINE__)

namespace Computation {
#ifndef __CUDACC__
    // Helper classes...
    struct BadCudaAlloc : public std::bad_alloc {
        explicit BadCudaAlloc(const char* what) NOEXCEPT
            : message(what) {}

        const char* what() const NOEXCEPT override {
            return message;
        }
    private:
        const char* message;
    };
#endif

    // Helper functions...
    void checkCudaError(cudaError err, const char* file, int line);
    void checkForDeviceMemoryLeaks();

    void* mallocDevice(size_t size);
    void* reallocDevice(void* ptr, size_t size);
    void freeDevice(void* ptr);

    void moveHostToDevice(void* target, const void* source, size_t count);
    void moveDeviceToHost(void* target, const void* source, size_t count);
    void moveDeviceToDevice(void* target, const void* source, size_t count);
    void fillValue(void* ptr, int value, size_t count);

    void deviceSync();

    void registerGraphicsResource(cudaGraphicsResource** graphicsResource, unsigned int bufferID);
    void unregisterGraphicsResource(cudaGraphicsResource* graphicsResource);
    void* mapGraphicsResource(cudaGraphicsResource** graphicsResource);
    void unmapGraphicsResource(cudaGraphicsResource* graphicsResource);

    void startCudaTimer(cudaEvent_t& start, cudaEvent_t& stop);
    float stopCudaTimer(cudaEvent_t& start, cudaEvent_t& stop);
}

#endif
