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
#include "SPHSolver.hpp"
#include "CudaHelper.hpp"
#include "SolverManager.hpp"

#include "SPHKernel.cuh"

namespace Computation {
    REGISTER_METATYPE(SPHSolverPtr);

    void SPHSolver::initialize(SolverManager* manager) {
        solverManager = manager;
    }

    void SPHSolver::cleanup() {

    }

    void SPHSolver::compute() {
        launchSPHKernel(solverManager->getParticleData(), solverManager->getGridData());
    }

    void SPHSolver::handleExternalForces(const QVector3D& source) {
        solverManager->bindData();

        Data::ParticleData* particleData = solverManager->getParticleData();
        computeExternalForcesHost(particleData, make_float4(source.x(), source.y(), source.z(), 0));

        solverManager->unbindData();
    }

    void SPHSolver::updateProperties(Data::SimulationConfig* config) {
        copyPropertiesToSPHKernel(config->getPropertyStore());
    }
}
