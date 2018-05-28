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

#include "Common.hpp"
#include "GridKernel.cuh"
#include "KernelHelper.cuh"
#include "CudaHelper.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace Computation {
#if USE_TEXTURE_MEMORY
    texture<float4, cudaTextureType1D> positionTex;
    texture<float4, cudaTextureType1D> velocityTex;
#endif

    __constant__
    Data::PropertyStore devicePropsGrid;
    Data::PropertyStore hostPropsGrid;

    __global__ void computeHashKeys(Data::ParticleData particleData, Data::GridData gridData);
    __global__ void reorder(Data::ParticleData particleData, Data::GridData gridData);

    void sort(Data::GridData* gridData);

    void copyPropertiesToGridKernel(const Data::PropertyStore& properties) {
        hostPropsGrid = properties;
        COPY_TO_SYMBOL_ASYNC(devicePropsGrid, &hostPropsGrid, sizeof(hostPropsGrid));
    }

    void launchGridKernel(Data::ParticleData* particleData, Data::GridData* gridData) {
        int count = hostPropsGrid.generalProperties.numParticles;
        int cellCount = hostPropsGrid.computeProperties.numCells;

        callCudaKernel(computeHashKeys, count, *particleData, *gridData);

        sort(gridData);

        fillValue(gridData->cellStart, 0xffffffff, cellCount * sizeof(unsigned int));

#if USE_TEXTURE_MEMORY
        BIND_TEXTURE(positionTex, particleData->position, count * sizeof(float4));
        BIND_TEXTURE(velocityTex, particleData->velocity, count * sizeof(float4));
#endif

        int blockSize, minGridSize;
        int sharedMemSize = sizeof(unsigned int);

        CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, reorder, sharedMemSize, count));

        int sharedMemBytes = sharedMemSize * (blockSize + 1);
        int gridSize = (count + blockSize - 1) / blockSize;

        reorder<<<gridSize, blockSize, sharedMemBytes>>>(*particleData, *gridData);

#if USE_TEXTURE_MEMORY
        UNBIND_TEXTURE(positionTex);
        UNBIND_TEXTURE(velocityTex);
#endif
    }

    __global__ void computeHashKeys(Data::ParticleData particleData, Data::GridData gridData) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= devicePropsGrid.generalProperties.numParticles) {
            return;
        }

        volatile float4 pos = particleData.position[tid];

        Data::ComputeProperties computeProps = devicePropsGrid.computeProperties;

        float3 worldSizeHalf = make_float3(computeProps.worldSizeHalfX,
            computeProps.worldSizeHalfY, computeProps.worldSizeHalfZ);

        float3 position3D = make_float3(pos.x, pos.y, pos.z);

        int3 gridPos = computeGridPosition(position3D,
            worldSizeHalf, computeProps.cellSize);

        unsigned int hash = computeGridHash(gridPos,
            devicePropsGrid.generalProperties.gridSize);

        gridData.hash[tid] = hash;
        gridData.index[tid] = tid;
    }

    __global__ void reorder(Data::ParticleData particleData, Data::GridData gridData) {
        extern __shared__ unsigned int sharedHash[];

        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= devicePropsGrid.generalProperties.numParticles) {
            return;
        }

        unsigned int hash = gridData.hash[tid];

        sharedHash[threadIdx.x + 1] = hash;

        if (tid > 0 && threadIdx.x == 0) {
            sharedHash[0] = gridData.hash[tid - 1];
        }

        __syncthreads();

        if (tid == 0 || hash != sharedHash[threadIdx.x]) {
            gridData.cellStart[hash] = tid;

            if (tid > 0) {
                gridData.cellEnd[sharedHash[threadIdx.x]] = tid;
            }
        }

        if (tid == devicePropsGrid.generalProperties.numParticles - 1) {
            gridData.cellEnd[hash] = tid + 1;
        }

        unsigned int index = gridData.index[tid];

        float4 pos = GET_VALUE(particleData, position, index);
        float4 vel = GET_VALUE(particleData, velocity, index);

        particleData.sortedPosition[tid] = pos;
        particleData.sortedVelocity[tid] = vel;
    }

    void sort(Data::GridData* gridData) {
        unsigned int particleCount = hostPropsGrid.generalProperties.numParticles;

        thrust::device_ptr<unsigned int> gridHashStart(gridData->hash.get());
        thrust::device_ptr<unsigned int> gridHashEnd(gridData->hash.get() + particleCount);
        thrust::device_ptr<unsigned int> gridIndices(gridData->index.get());

        thrust::sort_by_key(gridHashStart, gridHashEnd, gridIndices);
    }
}
