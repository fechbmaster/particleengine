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
#include "WaterBlockInteraction.hpp"
#include "AbstractInteractionManager.hpp"

#include "Utils/RandomSet.hpp"

namespace IO {
    REGISTER_METATYPE(WaterBlockInteractionPtr);

    WaterBlockInteraction::WaterBlockInteraction(QObject* parent)
        : KeyInteraction(Qt::Key_R, parent) {

    }

    void WaterBlockInteraction::initialize(AbstractInteractionManager* mgr) {

    }

    void WaterBlockInteraction::cleanup() {

    }

    void WaterBlockInteraction::updateProperties(Data::SimulationConfig* config) {
        simulationConfig = config;
    }

    void WaterBlockInteraction::process() {
        Utils::RandomSet<float> randomSet(particleCount * 4);
        randomSet.create(-0.1f, +0.1f);

        auto generalProps = simulationConfig->getGeneralProperties();
        generalProps.numParticles += particleCount;

        QVector<float> points = randomSet.toVector();
        for (int i = 1; i < randomSet.size(); i += 4) {
            points[i] += generalProps.worldSizeY; // increment Y axis
        }

        extendParticleBuffer(points.data(), points.size());

        simulationConfig->setGeneralProperties(generalProps);
        insertPending = false;
    }

    void WaterBlockInteraction::extendParticleBuffer(const float* points, int size) const {
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
}
