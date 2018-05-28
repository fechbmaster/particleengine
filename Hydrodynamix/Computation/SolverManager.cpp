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
#include "SolverManager.hpp"

#include "CudaHelper.hpp"

namespace Computation {
    void SolverManager::initialize() {
        cudaSetDevice(0);

        for (auto& solver : solvers) {
            solver->initialize(this);
        }
    }

    void SolverManager::cleanup() {
        for (auto& solver : solvers) {
            solver->cleanup();
        }

        setParticleBuffer(nullptr);
    }

    void SolverManager::compute() {
        bindData();

        for (auto& solver : solvers) {
            solver->compute();
        }

        unbindData();
        deviceSync();
    }

    void SolverManager::setParticleBuffer(QOpenGLBuffer* buffer) {
        particleBuffer = buffer;

        if (graphicsResource) {
            unregisterGraphicsResource(graphicsResource);
            graphicsResource = nullptr;
        }

        if (particleBuffer) {
            registerGraphicsResource(&graphicsResource, buffer->bufferId());
        }
    }

    void SolverManager::addSolver(const AbstractSolverPtr& solver) {
        solvers.append(solver);
    }

    void SolverManager::bindData() {
        void* mappedPtr = mapGraphicsResource(&graphicsResource);
        particleData.position.reset((float4*) mappedPtr);
    }

    void SolverManager::unbindData() {
        unmapGraphicsResource(graphicsResource);
        particleData.position.release();
    }

    void SolverManager::reallocateParticleData() {
        auto generalProps = simulationConfig->getGeneralProperties();

        uint particleCount = generalProps.numParticles;

        reallocCudaBuffer(particleData.velocity, particleCount);
        reallocCudaBuffer(particleData.force, particleCount);
        reallocCudaBuffer(particleData.surfaceNormal, particleCount);
        reallocCudaBuffer(particleData.density, particleCount);
        reallocCudaBuffer(particleData.colorField, particleCount);

        reallocCudaBuffer(particleData.sortedPosition, particleCount);
        reallocCudaBuffer(particleData.sortedVelocity, particleCount);
        reallocCudaBuffer(particleData.sortedDensity, particleCount);

        setParticleBuffer(particleBuffer);
    }

    void SolverManager::reallocateGridData() {
        auto generalProps = simulationConfig->getGeneralProperties();
        auto computeProps = simulationConfig->getComputeProperties();

        uint particleCount = generalProps.numParticles;

        reallocCudaBuffer(gridData.hash, particleCount);
        reallocCudaBuffer(gridData.index, particleCount);

        uint cellCount = computeProps.numCells;

        reallocCudaBuffer(gridData.cellStart, cellCount);
        reallocCudaBuffer(gridData.cellEnd, cellCount);
    }

    void SolverManager::updateProperties(Data::SimulationConfig* config) {
        simulationConfig = config;

        connect(config, &Data::SimulationConfig::generalPropertiesChanged, this, &SolverManager::propertiesChanged);
        connect(config, &Data::SimulationConfig::physicalPropertiesChanged, this, &SolverManager::propertiesChanged);

        propertiesChanged(config->getPropertyStore());
    }

    void SolverManager::updateComputeProperties(const Data::PropertyStore& props) const {
        Data::GeneralProperties generalProps = props.generalProperties;
        Data::PhysicalProperties physicalProps = props.physicalProperties;
        Data::ComputeProperties computeProps = props.computeProperties;

        computeProps.worldSizeHalfX = generalProps.worldSizeX / 2.0f;
        computeProps.worldSizeHalfY = generalProps.worldSizeY / 2.0f;
        computeProps.worldSizeHalfZ = generalProps.worldSizeZ / 2.0f;

        computeProps.cellSize = generalProps.particleRadius * 2;
        computeProps.numCells = pow(generalProps.gridSize, 3);

        float currentVolume = physicalProps.volume * (generalProps.numParticles / 50000.0f);

        computeProps.mass = (physicalProps.restDensity * currentVolume) / generalProps.numParticles;

        computeProps.surfaceTensionThreshold = sqrtf(physicalProps.restDensity / physicalProps.kernelParticles);
        computeProps.surfaceTensionThreshold2 = pow(computeProps.surfaceTensionThreshold, 2);

        computeProps.smoothingLength = powf((3.0f * currentVolume *  physicalProps.kernelParticles)
            / (4 * M_PI * generalProps.numParticles), 0.333f);
        computeProps.smoothingLength2 = pow(computeProps.smoothingLength, 2);
        computeProps.smoothingLength3 = pow(computeProps.smoothingLength, 3);

        computeProps.defaultKernelCoefficient = 315.0f / (64.0f * M_PI * pow(computeProps.smoothingLength, 9));
        computeProps.defaultKernelGradientCoefficient = -945.0f / (32.0f * M_PI * pow(computeProps.smoothingLength, 9));
        computeProps.defaultKernelLaplacianCoefficient = -945.0f / (32.0f * M_PI * pow(computeProps.smoothingLength, 9));

        computeProps.pressureKernelCoefficient = 15.0f / (M_PI * pow(computeProps.smoothingLength, 6));
        computeProps.pressureKernelGradientCoefficient = -45.0f / (M_PI * pow(computeProps.smoothingLength, 6));
        computeProps.pressureKernelLaplacianCoefficient = -90.0f / (M_PI * pow(computeProps.smoothingLength, 6));

        computeProps.viscosityKernelCoefficient = 15.0f / (2 * M_PI * pow(computeProps.smoothingLength, 3));
        computeProps.viscosityKernelGradientCoefficient = 15.0f / (2 * M_PI * pow(computeProps.smoothingLength, 3));
        computeProps.viscosityKernelLaplacianCoefficient = 45.0f / (M_PI * pow(computeProps.smoothingLength, 6));

        simulationConfig->setComputeProperties(computeProps);
    }

    void SolverManager::propertiesChanged(const Data::PropertyStore& properties) {
        updateComputeProperties(properties);

        reallocateParticleData();
        reallocateGridData();

        for (auto& solver : solvers) {
            solver->updateProperties(simulationConfig);
        }
    }
}
