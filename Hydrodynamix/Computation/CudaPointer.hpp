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

#ifndef CUDA_POINTER_HPP
#define CUDA_POINTER_HPP

#ifndef __CUDACC__
#include "CudaHelper.hpp"
#endif

namespace Computation {
    // A smart pointer wrapper class around an arbitrary cuda pointer
    // For nvcc this is just a POD structure with data access members
    template <class T>
    struct CudaPointer {
#ifndef __CUDACC__
        __host__
        explicit CudaPointer(T* obj = nullptr)
            : object(obj) {
        }

        __host__
        CudaPointer(CudaPointer&& r)
            : object(r.object) {

            r.object = nullptr;
        }

        template <class Y> __host__
        CudaPointer(CudaPointer<Y>&& r)
            : object(r.object) {

            r.object = nullptr;
        }

        __host__
        CudaPointer& operator = (CudaPointer&& r) {
            reset(r.object);
            r.object = nullptr;
            return *this;
        }

        template <class Y> __host__
        CudaPointer& operator = (CudaPointer<Y>&& r) {
            reset(r.object);
            r.object = nullptr;
            return *this;
        }

        __host__
        CudaPointer& operator = (std::nullptr_t) {
            reset(nullptr);
            return *this;
        }

        __host__
        ~CudaPointer() {
            reset(nullptr);
        }

        __host__
        void reset(T* obj) {
            if (object) {
                freeDevice(object);
            }
            object = obj;
        }

        __host__
        T* release() {
            T* obj = object;
            object = nullptr;
            return obj;
        }

        __host__
        void swap(CudaPointer& other) {
            T* tmp = object;
            object = other.object;
            other.object = tmp;
        }
#else
        __device__
        T* operator -> () {
            return object;
        }

        __device__
        const T* operator -> () const {
            return object;
        }

        __device__
        T& operator * () {
            return object;
        }

        __device__
        const T& operator * () const {
            return object;
        }

        __device__
        T& operator [] (size_t i) {
            return object[i];
        }

        __device__
        const T& operator [] (size_t i) const {
            return object[i];
        }
#endif

        __host__ __device__
        operator T* () {
            return object;
        }

        __host__ __device__
        operator const T* () const {
            return object;
        }

        __host__ __device__
        T* get() {
            return object;
        }

        __host__ __device__
        const T* get() const {
            return object;
        }

        __host__ __device__
        explicit operator bool() const {
            return object != nullptr;
        }

    private:
#ifndef __CUDACC__
        CudaPointer(const CudaPointer&);
        CudaPointer& operator = (const CudaPointer&);
#else
        CudaPointer();
#endif

    private:
        T* object;
    };

#ifndef __CUDACC__
    template <class T>
    CudaPointer<T> makeCudaBuffer(size_t count) {
        void* ptr = mallocDevice(count * sizeof(T));
        if (!ptr) {
            throw BadCudaAlloc("Failed to allocate device memory!");
        }
        return CudaPointer<T>((T*) ptr);
    }

    template <class T>
    void reallocCudaBuffer(CudaPointer<T>& ptr, size_t count) {
        void* obj = reallocDevice(ptr, count * sizeof(T));
        if (!obj) {
            throw BadCudaAlloc("Failed to reallocate device memory!");
        }
        ptr.release();
        ptr.reset((T*) obj);
    }
#endif
}

#endif
