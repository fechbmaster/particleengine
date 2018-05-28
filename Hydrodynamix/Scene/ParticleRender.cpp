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
#include "ParticleRender.hpp"
#include "AbstractScene.hpp"

#include "Utils/RandomSet.hpp"
#include "Data/SimulationConfig.hpp"

namespace Scene {
    REGISTER_METATYPE(ParticleRenderPtr);

    ParticleRender::ParticleRender(QObject* parent)
        : AbstractRender(parent) {

    }

    void ParticleRender::initialize(AbstractScene* parent) {
        properties = parent->getProperties();

        initializeOpenGLFunctions();
        shaderCollection.initPrograms(parent);

        // Enable gl_PointSize in vertex shader
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

        // If our window is resized we need to recreate the renderbuffers
        connect(parent, &AbstractScene::sizeChanged, this, &ParticleRender::sizeChanged);

        initParticles();
        initScreenQuad();
    }

    void ParticleRender::initParticles() {
        auto generalProps = properties->getGeneralProperties();

        uint32_t particleCount = generalProps.numParticles;
        float spawnRange = generalProps.worldSizeX / 4.0f;

        Utils::RandomSet<float> randomSet(particleCount * 4);
        randomSet.create(-spawnRange, +spawnRange);

        particleVAO.create();
        QOpenGLVertexArrayObject::Binder binder(&particleVAO);

        particleBuffer.create();
        particleBuffer.bind();
        particleBuffer.allocate(randomSet.data(), randomSet.size() * sizeof(float));
        shaderCollection.bindParticleBuffer(&particleBuffer);
    }

    void ParticleRender::initScreenQuad() {
        static const QVector3D quadData[] = {
            { -1.0f, -1.0f, 0.0f },
            { +1.0f, -1.0f, 0.0f },
            { -1.0f, +1.0f, 0.0f },
            { -1.0f, +1.0f, 0.0f },
            { +1.0f, -1.0f, 0.0f },
            { +1.0f, +1.0f, 0.0f }
        };

        quadVAO.create();
        QOpenGLVertexArrayObject::Binder binder(&quadVAO);

        quadBuffer.create();
        quadBuffer.bind();
        quadBuffer.allocate(quadData, sizeof(quadData));
        shaderCollection.bindScreenQuadBuffer(&quadBuffer);
    }

    void ParticleRender::initBuffers(int width, int height) {
        // Low-resolution textures
        int widthLow = width / 2;
        int heightLow = height / 2;

        glGenRenderbuffers(1, &depthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, widthLow, heightLow);

        makeBufferForPass(RenderPasses::Depth, widthLow, heightLow, GL_RG, GL_RG32F, depthBuffer);
        makeBufferForPass(RenderPasses::DepthBack, widthLow, heightLow, GL_RG, GL_RG32F, 0);
        makeBufferForPass(RenderPasses::Thickness, widthLow, heightLow, GL_RED, GL_R8, 0);
        makeBufferForPass(RenderPasses::ThicknessBack, widthLow, heightLow, GL_RED, GL_R8, 0);

        buffersInitialized = true;
    }

    void ParticleRender::freeBuffers() {
        if (!buffersInitialized) {
            return;
        }

        glDeleteFramebuffers(RenderPasses::Count, frameBuffers);
        glDeleteTextures(RenderPasses::Count, renderTextures);
        glDeleteRenderbuffers(1, &depthBuffer);

        buffersInitialized = false;
    }

    void ParticleRender::makeBufferForPass(int pass, int width,
        int height, GLenum format, GLint internalFormat, GLuint depth) {

        frameBuffers[pass] = makeFramebuffer(width, height,
            format, internalFormat, &renderTextures[pass], depth);
    }

    void ParticleRender::sizeChanged(int width, int height) {
        freeBuffers();
        initBuffers(width, height);
    }

    void ParticleRender::cleanup() {
        freeBuffers();
    }

    void ParticleRender::update() {

    }

    void ParticleRender::render() {
        // Default draw mode (no surface rendering)
        if (renderingMode == RenderingMode::RenderSpheres) {
            renderParticles(ParticlePrograms::Default);
            return;
        }

        // Backup previous rendertarget (for switch to offscreen)
        GLint frameBuffer;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &frameBuffer);

        // Switch to low resolution viewport
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        glViewport(vp[0], vp[1], vp[2] / 2, vp[3] / 2);

        // We'll be clearing the buffers to 0
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // Render depth, thickness & curvature in lower resolution
        renderDepthPass();
        renderThicknessPass();
        renderBlurPass();
        renderCurvaturePass();

        // Switch back to high-res viewport & rendertarget
        glViewport(vp[0], vp[1], vp[2], vp[3]);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

        // Render final image
        renderColorPass();
    }

    // Renders all the particles with the given shader program
    void ParticleRender::renderParticles(uint programId) {
        auto generalProps = properties->getGeneralProperties();

        shaderCollection.getProgram(programId)->bind();
        {
            QOpenGLVertexArrayObject::Binder binder(&particleVAO);
            glDrawArrays(GL_POINTS, 0, generalProps.numParticles);
        }
        shaderCollection.getProgram(programId)->release();
    }

    // Renders every fragment of the screen with the given shader program
    void ParticleRender::renderScreenQuad(uint programId) {
        shaderCollection.getProgram(programId)->bind();
        {
            QOpenGLVertexArrayObject::Binder binder(&quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        shaderCollection.getProgram(programId)->release();
    }

    // Renders particle depth
    void ParticleRender::renderDepthPass() {
        // Switch to depth framebuffer
        GLuint frameBuffer = frameBuffers[RenderPasses::Depth];
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw particle depth
        renderParticles(ParticlePrograms::Depth);

        // Depth is not needed anymore (this is good for performance)
        GLenum attachment = GL_DEPTH_ATTACHMENT;
        glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &attachment);
    }

    // Renders particle thickness (for light refraction)
    void ParticleRender::renderThicknessPass() {
        // Switch to thickness framebuffer
        GLuint frameBuffer = frameBuffers[RenderPasses::Thickness];
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glDisable(GL_DEPTH_TEST);

        // Draw particle thickness (additive blend, no depth test)
        renderParticles(ParticlePrograms::Thickness);

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
    }

    // Blurs the thickness texture to make thickness look smooth
    void ParticleRender::renderBlurPass() {
        glDisable(GL_DEPTH_TEST);
        glActiveTexture(GL_TEXTURE0);

        for (uint i = 0; i < 2; ++i) {
            // Swap buffers in each iteration
            GLuint frameBuffer = frameBuffers[RenderPasses::Thickness + (i + 1) % 2];
            GLuint textureBuffer = renderTextures[RenderPasses::Thickness + (i % 2)];

            glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
            glBindTexture(GL_TEXTURE_2D, textureBuffer);

            // Swap blur direction in each iteration (horizontal, vertical)
            shaderCollection.setBlurDirection(QVector2D((i + 1) % 2, i % 2));
            renderScreenQuad(ParticlePrograms::Blur);
        }

        glEnable(GL_DEPTH_TEST);
    }

    // Applies curvature flow smoothing to our depth buffers
    void ParticleRender::renderCurvaturePass() {
        glDisable(GL_DEPTH_TEST);
        glActiveTexture(GL_TEXTURE0);

        for (uint i = 0; i < smoothingIterations * 2; ++i) {
            // Swap depth / front buffer in each iteration
            GLuint textureBuffer = renderTextures[RenderPasses::Depth + (i % 2)];
            GLuint frameBuffer = frameBuffers[RenderPasses::Depth + (i + 1) % 2];

            glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
            glBindTexture(GL_TEXTURE_2D, textureBuffer);

            // Draw next iteration (using curvature flow program)
            renderScreenQuad(ParticlePrograms::CurvatureFlow);
        }

        glEnable(GL_DEPTH_TEST);
    }

    // Renders the final composition (liquid shader)
    void ParticleRender::renderColorPass() {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderTextures[RenderPasses::Depth]);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, renderTextures[RenderPasses::Thickness]);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);

        // Draw final image (using liquid program, no depth write)
        renderScreenQuad(ParticlePrograms::Liquid);

        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
}
