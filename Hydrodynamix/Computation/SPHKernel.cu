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
#include "SPHKernel.cuh"
#include "KernelHelper.cuh"
#include "CudaHelper.hpp"
#include "Operators.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

namespace Computation {
#if USE_TEXTURE_MEMORY
    texture<float4, cudaTextureType1D> sortedPositionTex;
    texture<float4, cudaTextureType1D> sortedVelocityTex;
    texture<float, cudaTextureType1D> sortedDensityTex;

    texture<unsigned int, cudaTextureType1D> cellStartTex;
    texture<unsigned int, cudaTextureType1D> cellEndTex;
#endif

    __constant__
    Data::PropertyStore devicePropsSPH;
    Data::PropertyStore hostPropsSPH;

    __global__ void computeDensity(Data::ParticleData particleData, Data::GridData gridData);
    __global__ void computeInternalForces(Data::ParticleData particleData, Data::GridData gridData);
    __global__ void integrate(Data::ParticleData particleData);
    __global__ void boundary(Data::ParticleData particleData);

    __global__ void computeExternalForces(Data::ParticleData particleData, float4 source);

    // Poly6 Kernels
    __device__ float defaultKernel(float dist, const Data::ComputeProperties& computeProps);
    __device__ float4 defaultKernelGradient(float dist, const float4& r, const Data::ComputeProperties& computeProps);
    __device__ float defaultKernelLaplacian(float dist, const Data::ComputeProperties& computeProps);

    // Spiky Kernels
    __device__ float pressureKernel(float dist, const Data::ComputeProperties& computeProps);
    __device__ float4 pressureKernelGradient(float dist, const float4& r, const Data::ComputeProperties& computeProps);
    __device__ float pressureKernelLaplacian(float dist, const Data::ComputeProperties& computeProps);

    __device__ float viscosityKernel(float dist, const Data::ComputeProperties& computeProps);
    __device__ float4 viscosityKernelGradient(float dist, const float4& r, const Data::ComputeProperties& computeProps);
    __device__ float viscosityKernelLaplacian(float dist, const Data::ComputeProperties& computeProps);

    void copyPropertiesToSPHKernel(const Data::PropertyStore& properties) {
        hostPropsSPH = properties;
        COPY_TO_SYMBOL_ASYNC(devicePropsSPH, &hostPropsSPH, sizeof(hostPropsSPH));
    }

    void launchSPHKernel(Data::ParticleData* particleData, Data::GridData* gridData) {
        int count = hostPropsSPH.generalProperties.numParticles;

#if USE_TEXTURE_MEMORY
        int cellCount = hostPropsSPH.computeProperties.numCells;
        BIND_TEXTURE(sortedPositionTex, particleData->sortedPosition, count * sizeof(float4));
        BIND_TEXTURE(sortedVelocityTex, particleData->sortedVelocity, count * sizeof(float4));
        BIND_TEXTURE(sortedDensityTex, particleData->sortedDensity, count * sizeof(float));
        BIND_TEXTURE(cellStartTex, gridData->cellStart, cellCount * sizeof(unsigned int));
        BIND_TEXTURE(cellEndTex, gridData->cellEnd, cellCount * sizeof(unsigned int));
#endif

        callCudaKernel(computeDensity, count, *particleData, *gridData);
        callCudaKernel(computeInternalForces, count, *particleData, *gridData);

#if USE_TEXTURE_MEMORY
        UNBIND_TEXTURE(sortedPositionTex);
        UNBIND_TEXTURE(sortedVelocityTex);
        UNBIND_TEXTURE(sortedDensityTex);
        UNBIND_TEXTURE(cellStartTex);
        UNBIND_TEXTURE(cellEndTex);
#endif

        callCudaKernel(integrate, count, *particleData);
        callCudaKernel(boundary, count, *particleData);
    }

    void computeExternalForcesHost(Data::ParticleData* particleData, const float4& source) {
        int count = hostPropsSPH.generalProperties.numParticles;
        callCudaKernel(computeExternalForces, count, *particleData, source);
    }

    __global__ void computeDensity(Data::ParticleData particleData, Data::GridData gridData) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= devicePropsSPH.generalProperties.numParticles) {
            return;
        }

        Data::ComputeProperties computeProps = devicePropsSPH.computeProperties;
        float density = defaultKernel(0, computeProps);

        float4 posA = GET_VALUE(particleData, sortedPosition, tid);

        float3 worldSizeHalf = make_float3(computeProps.worldSizeHalfX,
            computeProps.worldSizeHalfY, computeProps.worldSizeHalfZ);

        float3 position3D = make_float3(posA.x, posA.y, posA.z);

        int3 gridPos = computeGridPosition(position3D,
            worldSizeHalf, computeProps.cellSize);

        unsigned int gridSize = devicePropsSPH.generalProperties.gridSize;

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    int3 neighbourGridPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);

                    unsigned int gridHash = computeGridHash(neighbourGridPos, gridSize);
                    unsigned int startIndex = GET_VALUE(gridData, cellStart, gridHash);

                    if (startIndex == 0xffffffff) {
                        continue;
                    }

                    unsigned int endIndex = GET_VALUE(gridData, cellEnd, gridHash);

                    for (unsigned int i = startIndex; i < endIndex; i++) {
                        if (i == tid) {
                            continue;
                        }

                        float4 posB = GET_VALUE(particleData, sortedPosition, i);
                        float dist2 = lensq3D(posA - posB);

                        if (dist2 < computeProps.smoothingLength2 && dist2 > 0) {
                            density += defaultKernel(dist2, computeProps);
                        }
                    }
                }
            }
        }

        density *= computeProps.mass * computeProps.defaultKernelCoefficient;

        unsigned int index = gridData.index[tid];
        particleData.density[index] = density;
        particleData.sortedDensity[tid] = density;
    }

    __global__ void computeInternalForces(Data::ParticleData particleData, Data::GridData gridData) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= devicePropsSPH.generalProperties.numParticles) {
            return;
        }

        float4 pressureForce = make_float4(0, 0, 0, 0);
        float4 viscosityForce = make_float4(0, 0, 0, 0);
        float4 surfaceNormal = make_float4(0, 0, 0, 0);
        float colorField = 0.0f;

        float4 posA = GET_VALUE(particleData, sortedPosition, tid);
        float4 velA = GET_VALUE(particleData, sortedVelocity, tid);
        float densityA = GET_VALUE(particleData, sortedDensity, tid);

        Data::PropertyStore deviceProps = devicePropsSPH;
        float gasStiffness = deviceProps.physicalProperties.gasStiffness;
        float restDensity = deviceProps.physicalProperties.restDensity;
        float particleMass = deviceProps.computeProperties.mass;

        float pressureA = gasStiffness * (densityA - restDensity);

        float3 worldSizeHalf = make_float3(
            deviceProps.computeProperties.worldSizeHalfX,
            deviceProps.computeProperties.worldSizeHalfY,
            deviceProps.computeProperties.worldSizeHalfZ);

        float3 position3D = make_float3(posA.x, posA.y, posA.z);

        int3 gridPos = computeGridPosition(position3D,
            worldSizeHalf, deviceProps.computeProperties.cellSize);

        unsigned int gridSize = deviceProps.generalProperties.gridSize;

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    int3 neighbourGridPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);

                    unsigned int gridHash = computeGridHash(neighbourGridPos, gridSize);
                    unsigned int startIndex = GET_VALUE(gridData, cellStart, gridHash);

                    if (startIndex == 0xffffffff) {
                        continue;
                    }

                    unsigned int endIndex = GET_VALUE(gridData, cellEnd, gridHash);

                    for (unsigned int i = startIndex; i < endIndex; i++) {
                        if (i == tid) {
                            continue;
                        }

                        float4 posB = GET_VALUE(particleData, sortedPosition, i);
                        float4 velB = GET_VALUE(particleData, sortedVelocity, i);
                        float densityB = GET_VALUE(particleData, sortedDensity, i);
                        float pressureB = gasStiffness * (densityB - restDensity);

                        float4 deltaPos = posA - posB;
                        float dist2 = lensq3D(deltaPos);

                        if (dist2 >= deviceProps.computeProperties.smoothingLength2 || dist2 <= 0) {
                            continue;
                        }

                        float dist = sqrtf(dist2);

                        pressureForce += ((pressureA + pressureB) / (2.0f * densityB))
                            * pressureKernelGradient(dist, deltaPos, deviceProps.computeProperties);
                        viscosityForce += ((velB - velA) / densityB) * viscosityKernelLaplacian(dist, deviceProps.computeProperties);
                        surfaceNormal += defaultKernelGradient(dist2, deltaPos, deviceProps.computeProperties) / densityB;
                        colorField += defaultKernelLaplacian(dist2, deviceProps.computeProperties) / densityB;
                    }
                }
            }
        }

        pressureForce *= -particleMass * deviceProps.computeProperties.pressureKernelGradientCoefficient;
        viscosityForce *= particleMass * deviceProps.physicalProperties.viscosity
            * deviceProps.computeProperties.viscosityKernelLaplacianCoefficient;
        surfaceNormal *= particleMass * deviceProps.computeProperties.defaultKernelGradientCoefficient;
        colorField *= particleMass * deviceProps.computeProperties.defaultKernelLaplacianCoefficient;

        unsigned int index = gridData.index[tid];
        particleData.force[index] = pressureForce + viscosityForce;
        particleData.surfaceNormal[index] = surfaceNormal;
        particleData.colorField[index] = colorField;
    }

    __global__ void integrate(Data::ParticleData particleData) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= devicePropsSPH.generalProperties.numParticles) {
            return;
        }

        float4 pos = particleData.position[tid];
        float4 vel = particleData.velocity[tid];
        float4 force = particleData.force[tid];
        float4 surfaceNormal = particleData.surfaceNormal[tid];

        float density = particleData.density[tid];
        float colorField = particleData.colorField[tid];
        float timeStep = devicePropsSPH.physicalProperties.timeStep;

        force.y -= devicePropsSPH.physicalProperties.gravity * density;

        float surfaceVel = 0.0f;
        float surfaceNormalLensq = lensq3D(surfaceNormal);

        if (surfaceNormalLensq >= devicePropsSPH.computeProperties.surfaceTensionThreshold2) {
            force -= devicePropsSPH.physicalProperties.surfaceTension
                * colorField * (surfaceNormal / sqrtf(surfaceNormalLensq));

            surfaceVel = lensq3D(vel);
        }

        // lerp foam color and surface velocity
        float foamFactor = timeStep * surfaceVel + (1.0f - timeStep) * pos.w;

        vel += force * timeStep / density;
        pos += vel * timeStep;

        pos.w = fmaxf(foamFactor, 0.1f);

        particleData.position[tid] = pos;
        particleData.velocity[tid] = vel;
    }

    __global__ void boundary(Data::ParticleData particleData) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= devicePropsSPH.generalProperties.numParticles) {
            return;
        }

        float4 pos = particleData.position[tid];
        float4 vel = particleData.velocity[tid];

        float worldSizeHalfX = devicePropsSPH.computeProperties.worldSizeHalfX;
        float worldSizeHalfY = devicePropsSPH.computeProperties.worldSizeHalfY;
        float worldSizeHalfZ = devicePropsSPH.computeProperties.worldSizeHalfZ;

        float radius = devicePropsSPH.generalProperties.particleRadius;
        float elasticity = devicePropsSPH.physicalProperties.elasticity;

        if (pos.x + radius > worldSizeHalfX) {
            pos.x = worldSizeHalfX - radius;
            vel.x *= -elasticity;
        } else if (pos.x - radius < -worldSizeHalfX) {
            pos.x = -worldSizeHalfX + radius;
            vel.x *= -elasticity;
        }

        if (pos.y - radius < -worldSizeHalfY) {
            pos.y = -worldSizeHalfY + radius;
            vel.y *= -elasticity;
        }

        if (pos.z + radius > worldSizeHalfZ) {
            pos.z = worldSizeHalfZ - radius;
            vel.z *= -elasticity;
        } else if (pos.z - radius < -worldSizeHalfZ) {
            pos.z = -worldSizeHalfZ + radius;
            vel.z *= -elasticity;
        }

        particleData.position[tid] = pos;
        particleData.velocity[tid] = vel;
    }

    __global__ void computeExternalForces(Data::ParticleData particleData, float4 source) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= devicePropsSPH.generalProperties.numParticles) {
            return;
        }

        float4 pos = particleData.position[tid];
        float4 vel = particleData.velocity[tid];

        float4 delta = source - pos;
        float dist = fmaxf(length3D(delta), 1.0f);
        vel += delta / dist * devicePropsSPH.physicalProperties.mouseForce * devicePropsSPH.physicalProperties.timeStep;

        particleData.velocity[tid] = vel;
    }

    __device__ float defaultKernel(float dist2, const Data::ComputeProperties& computeProps) {
        return pow(computeProps.smoothingLength2 - dist2, 3);
    }

    __device__ float4 defaultKernelGradient(float dist2, const float4& r, const Data::ComputeProperties& computeProps) {
        return r * pow(computeProps.smoothingLength2 - dist2, 2);
    }

    __device__ float defaultKernelLaplacian(float dist2, const Data::ComputeProperties& computeProps) {
        return (computeProps.smoothingLength2 - dist2) * (3 * computeProps.smoothingLength2 - 7 * dist2);
    }

    __device__ float pressureKernel(float dist, const Data::ComputeProperties& computeProps) {
        return pow(computeProps.smoothingLength - dist, 3);
    }

    __device__ float4 pressureKernelGradient(float dist, const float4& r, const Data::ComputeProperties& computeProps) {
        return pow(computeProps.smoothingLength - dist, 2) * (r / dist);
    }

    __device__ float pressureKernelLaplacian(float dist, const Data::ComputeProperties& computeProps) {
        return (computeProps.smoothingLength - dist) * (computeProps.smoothingLength - 2 * dist) * (1 / dist);
    }

    __device__ float viscosityKernel(float dist, const Data::ComputeProperties& computeProps) {
        return ((-dist * dist * dist) / (2 * computeProps.smoothingLength3)) + ((dist * dist)
                / computeProps.smoothingLength2) + (computeProps.smoothingLength / (2 * dist)) - 1;
    }

    __device__ float4 viscosityKernelGradient(float dist, const float4& r, const Data::ComputeProperties& computeProps) {
        return r * (-(3 * dist) / (2 * computeProps.smoothingLength3) + (2 / computeProps.smoothingLength2)
                - (computeProps.smoothingLength / (2 * dist * dist * dist)));
    }

    __device__ float viscosityKernelLaplacian(float dist, const Data::ComputeProperties& computeProps) {
        return computeProps.smoothingLength - dist;
    }
}
