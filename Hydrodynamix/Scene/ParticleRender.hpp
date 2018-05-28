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

#ifndef PARTICLE_RENDER_HPP
#define PARTICLE_RENDER_HPP

#include "AbstractRender.hpp"
#include "ParticleShaderCollection.hpp"
#include "Utils/OpenGLFunctions.hpp"

#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

namespace Scene {
    class ParticleRender : public AbstractRender, private Utils::OpenGLFunctions {
        Q_OBJECT

    public:
        enum RenderingMode {
            RenderSpheres,
            RenderSurface
        };

    public:
        ParticleRender(QObject* parent = nullptr);
        ~ParticleRender() override = default;

    public:
        void initialize(AbstractScene* parent) override;
        void cleanup() override;
        void update() override;
        void render() override;

        QOpenGLBuffer* getBuffer() {
            return &particleBuffer;
        }

    private:
        void initParticles();
        void initScreenQuad();

        void initBuffers(int width, int height);
        void freeBuffers();

        void makeBufferForPass(int pass, int width, int height,
            GLenum format, GLint internalFormat, GLuint depth);

        void renderParticles(uint programId);
        void renderScreenQuad(uint programId);

        void renderDepthPass();
        void renderThicknessPass();
        void renderBlurPass();
        void renderCurvaturePass();
        void renderColorPass();

    private slots:
        void sizeChanged(int w, int h);

    private:
        struct RenderPasses {
            enum RenderPass {
                Depth = 0,
                DepthBack,
                Thickness,
                ThicknessBack,
                Count
            };
        };

        GLuint depthBuffer = 0;
        GLuint frameBuffers[RenderPasses::Count];
        GLuint renderTextures[RenderPasses::Count];

        bool buffersInitialized = false;

    private:
        QOpenGLBuffer particleBuffer;
        QOpenGLBuffer quadBuffer;
        QOpenGLVertexArrayObject particleVAO;
        QOpenGLVertexArrayObject quadVAO;

        uint smoothingIterations = 40;

        ParticleShaderCollection shaderCollection;
        RenderingMode renderingMode = RenderSurface;

        Data::SimulationConfig* properties = nullptr;

    public:
        DEF_GETTER(getRenderingMode, renderingMode);
        DEF_SETTER(setRenderingMode, renderingMode);

        DEF_GETTER(getSmoothingIterations, smoothingIterations);
        DEF_SETTER(setSmoothingIterations, smoothingIterations);
    };

    typedef QSharedPointer<ParticleRender> ParticleRenderPtr;
}

Q_DECLARE_METATYPE(Scene::ParticleRenderPtr);

#endif
