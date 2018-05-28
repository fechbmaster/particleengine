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

#ifndef WATER_BLOCK_INTERACTION_HPP
#define WATER_BLOCK_INTERACTION_HPP

#include "KeyInteraction.hpp"

#include <QOpenGLBuffer>

namespace IO {
    class WaterBlockInteraction : public KeyInteraction {
        Q_OBJECT

    public:
        WaterBlockInteraction(QObject* parent = nullptr);
        ~WaterBlockInteraction() override = default;

    public:
        void initialize(AbstractInteractionManager*) override;
        void cleanup() override;
        void process() override;

        void updateProperties(Data::SimulationConfig* config) override;

    public:
        void start() override {
            insertPending = true;
        }

        void stop() override {
            insertPending = false;
        }

        bool isRunning() const override {
            return insertPending;
        }

    private:
        void extendParticleBuffer(const float* points, int size) const;

    private:
        uint particleCount = 500;
        bool insertPending = false;

        QOpenGLBuffer* particleBuffer = nullptr;
        Data::SimulationConfig* simulationConfig = nullptr;

    public:
        DEF_GETTER(getParticleBuffer, particleBuffer);
        DEF_SETTER(setParticleBuffer, particleBuffer);
    };

    typedef QSharedPointer<WaterBlockInteraction> WaterBlockInteractionPtr;
}

Q_DECLARE_METATYPE(IO::WaterBlockInteractionPtr);

#endif
