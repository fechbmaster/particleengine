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

#ifndef PARTICLE_SHADER_COLLECTION_HPP
#define PARTICLE_SHADER_COLLECTION_HPP

#include "AbstractShaderCollection.hpp"
#include "Data/SimulationConfig.hpp"

#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>

namespace Scene {
    struct ParticlePrograms {
        enum ParticleProgram {
            Default = 0,
            Depth,
            Thickness,
            Blur,
            CurvatureFlow,
            Liquid,
            Count
        };
    };

    class ParticleShaderCollection : public AbstractShaderCollection {
        Q_OBJECT

    public:
        void initPrograms(AbstractScene* parentScene) override;

    public:
        void bindParticleBuffer(QOpenGLBuffer* particleBuffer);
        void bindScreenQuadBuffer(QOpenGLBuffer* screenQuadBuffer);
        void setBlurDirection(const QVector2D& direction);

    public:
        QOpenGLShaderProgram* getProgram(uint programId) override {
            Q_ASSERT(programId < ParticlePrograms::Count);
            return &shaderPrograms[programId];
        }

    private:
        void initShaderUniforms(const Data::SimulationConfig* params);

    private:
        uint blurDirection = 0;
        QOpenGLShaderProgram shaderPrograms[ParticlePrograms::Count];
    };
}

#endif
