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
#include "WaterJetInteraction.hpp"
#include "AbstractInteractionManager.hpp"

#include "Utils/RandomSet.hpp"
#include "Computation/CudaHelper.hpp"
#include "Computation/KernelHelper.cuh"

#include <vector_types.h>
#include <vector_functions.h>

namespace IO {
    REGISTER_METATYPE(WaterJetInteractionPtr);

    WaterJetInteraction::WaterJetInteraction(QObject* parent)
        : KeyInteraction(Qt::Key_F, parent) {

        setInteractionType(KeyInteraction::HoldPressed);
    }

    void WaterJetInteraction::initialize(AbstractInteractionManager* mgr) {

    }

    void WaterJetInteraction::cleanup() {

    }

    void WaterJetInteraction::updateProperties(Data::SimulationConfig* config) {
        simulationConfig = config;
    }

    void WaterJetInteraction::process() {
        Utils::RandomSet<float> randomSet(particleCount * 4);
        randomSet.create(-0.1f, +0.0f);

        auto generalProps = simulationConfig->getGeneralProperties();
        uint bufferOffset = generalProps.numParticles;
        generalProps.numParticles += particleCount;

        QVector<float> points = randomSet.toVector();
        for (int i = 0; i < randomSet.size(); i += 4) {
            points[i] += generalProps.worldSizeX / 2;
        }

        extendParticleBuffer(points.data(), points.size());
        simulationConfig->setGeneralProperties(generalProps);

        QVector3D vel = calcVelocityToTarget(jetTarget);
        float4 velocity = make_float4(vel.x(), vel.y(), vel.z(), 0.0f);

        float4* velocityBuffer = particleData->velocity;
        Computation::fillFloat4Array(velocityBuffer + bufferOffset, particleCount, velocity);
    }

    void WaterJetInteraction::extendParticleBuffer(const float* points, int size) const {
        auto generalProps = simulationConfig->getGeneralProperties();

        int count = generalProps.numParticles * 4;
        int bufferSize = count * sizeof(float);

        QVector<float> bufferData;
        bufferData.resize(count);

        particleBuffer->bind();
        particleBuffer->read(0, bufferData.data(), bufferSize);

        bufferData.reserve(bufferData.size() + size);

        for (uint i = 0; i < size; ++i) {
            bufferData.append(points[i]);
        }

        bufferSize = bufferData.size() * sizeof(float);
        particleBuffer->allocate(bufferData.data(), bufferSize);
        particleBuffer->release();
    }

    QVector3D WaterJetInteraction::calcVelocityToTarget(const QVector3D& target) const {
        auto generalProps = simulationConfig->getGeneralProperties();
        auto physicalProps = simulationConfig->getPhysicalProperties();

        QVector3D position;
        position.setX(generalProps.worldSizeX / 2.0f);

        QVector3D targetVec = target - position;
        targetVec.setY(0.0f);

        float h = generalProps.worldSizeY / 2.0f;
        float t = sqrtf(h / (0.5f * physicalProps.gravity));
        return targetVec / t;
    }
}
