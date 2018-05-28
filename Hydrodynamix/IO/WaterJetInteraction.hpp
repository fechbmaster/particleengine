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

#ifndef WATER_JET_INTERACTION_HPP
#define WATER_JET_INTERACTION_HPP

#include "KeyInteraction.hpp"
#include "Data/ParticleData.hpp"

#include <QOpenGLBuffer>

namespace IO {
    class WaterJetInteraction : public KeyInteraction {
        Q_OBJECT

    public:
        WaterJetInteraction(QObject* parent = nullptr);
        ~WaterJetInteraction() override = default;

    public:
        void initialize(AbstractInteractionManager*) override;
        void cleanup() override;
        void process() override;

        void updateProperties(Data::SimulationConfig* config) override;

    public:
        void start() override {
            running = true;
        }

        void stop() override {
            running = false;
        }

        bool isRunning() const override {
            return running;
        }

    private:
        void extendParticleBuffer(const float* points, int size) const;
        QVector3D calcVelocityToTarget(const QVector3D& target) const;

    private:
        uint particleCount = 20;
        bool running = false;

        QVector3D jetTarget;

        QOpenGLBuffer* particleBuffer = nullptr;
        Data::ParticleData* particleData = nullptr;
        Data::SimulationConfig* simulationConfig = nullptr;

    public:
        DEF_GETTER(getParticleBuffer, particleBuffer);
        DEF_SETTER(setParticleBuffer, particleBuffer);

        DEF_GETTER(getParticleData, particleData);
        DEF_SETTER(setParticleData, particleData);

        DEF_GETTER(getJetTarget, jetTarget);
        DEF_SETTER(setJetTarget, jetTarget);
    };

    typedef QSharedPointer<WaterJetInteraction> WaterJetInteractionPtr;
}

Q_DECLARE_METATYPE(IO::WaterJetInteractionPtr);

#endif
