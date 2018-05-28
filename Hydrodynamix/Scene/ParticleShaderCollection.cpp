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
#include "ParticleShaderCollection.hpp"
#include "AbstractScene.hpp"

namespace Scene {
    void ParticleShaderCollection::initPrograms(AbstractScene* parentScene) {
        auto defaultShader = getProgram(ParticlePrograms::Default);
        defaultShader->addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ParticleVertex.glsl");
        defaultShader->addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ParticleFragment.glsl");
        defaultShader->link();

        parentScene->bindGlobalParams(defaultShader);

        auto depthShader = getProgram(ParticlePrograms::Depth);
        depthShader->addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ParticleVertex.glsl");
        depthShader->addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ParticleDepthFragment.glsl");
        depthShader->link();

        parentScene->bindGlobalParams(depthShader);

        auto thicknessShader = getProgram(ParticlePrograms::Thickness);
        thicknessShader->addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ParticleVertex.glsl");
        thicknessShader->addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ParticleThicknessFragment.glsl");
        thicknessShader->link();

        parentScene->bindGlobalParams(thicknessShader);

        auto blurShader = getProgram(ParticlePrograms::Blur);
        blurShader->addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ParticleScreenVertex.glsl");
        blurShader->addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ParticleBlurFragment.glsl");
        blurShader->link();

        parentScene->bindGlobalParams(blurShader);

        auto curvatureFlowShader = getProgram(ParticlePrograms::CurvatureFlow);
        curvatureFlowShader->addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ParticleScreenVertex.glsl");
        curvatureFlowShader->addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ParticleCurvatureFlowFragment.glsl");
        curvatureFlowShader->link();

        parentScene->bindGlobalParams(curvatureFlowShader);

        auto liquidShader = getProgram(ParticlePrograms::Liquid);
        liquidShader->addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ParticleScreenVertex.glsl");
        liquidShader->addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ParticleLiquidFragment.glsl");
        liquidShader->link();

        parentScene->bindGlobalParams(liquidShader);

        initShaderUniforms(parentScene->getProperties());
    }

    void ParticleShaderCollection::initShaderUniforms(const Data::SimulationConfig* config) {
        auto generalProps = config->getGeneralProperties();

        auto defaultShader = getProgram(ParticlePrograms::Default);
        defaultShader->bind();
        defaultShader->setUniformValue("radius", generalProps.particleRadius);
        defaultShader->release();

        auto depthShader = getProgram(ParticlePrograms::Depth);
        depthShader->bind();
        depthShader->setUniformValue("radius", generalProps.particleRadius);
        depthShader->release();

        auto thicknessShader = getProgram(ParticlePrograms::Thickness);
        thicknessShader->bind();
        thicknessShader->setUniformValue("radius", generalProps.particleRadius);
        thicknessShader->release();

        auto blurShader = getProgram(ParticlePrograms::Blur);
        blurShader->bind();
        blurShader->setUniformValue("blurTexture", 0);
        blurDirection = blurShader->uniformLocation("blurDirection");
        blurShader->release();

        auto curvatureFlowShader = getProgram(ParticlePrograms::CurvatureFlow);
        curvatureFlowShader->bind();
        curvatureFlowShader->setUniformValue("particleTexture", 0);
        curvatureFlowShader->release();

        auto liquidShader = getProgram(ParticlePrograms::Liquid);
        liquidShader->bind();
        liquidShader->setUniformValue("particleTexture", 0);
        liquidShader->setUniformValue("thicknessTexture", 1);
        liquidShader->release();
    }

    void ParticleShaderCollection::bindParticleBuffer(QOpenGLBuffer* particleBuffer) {
        particleBuffer->bind();

        auto defaultShader = getProgram(ParticlePrograms::Default);
        defaultShader->setAttributeBuffer("position0", GL_FLOAT, 0, 4);
        defaultShader->enableAttributeArray("position0");

        auto depthShader = getProgram(ParticlePrograms::Depth);
        depthShader->setAttributeBuffer("position0", GL_FLOAT, 0, 4);
        depthShader->enableAttributeArray("position0");

        auto thicknessShader = getProgram(ParticlePrograms::Thickness);
        thicknessShader->setAttributeBuffer("position0", GL_FLOAT, 0, 4);
        thicknessShader->enableAttributeArray("position0");
    }

    void ParticleShaderCollection::bindScreenQuadBuffer(QOpenGLBuffer* screenQuadBuffer) {
        screenQuadBuffer->bind();

        auto blurShader = getProgram(ParticlePrograms::Blur);
        blurShader->setAttributeBuffer("position0", GL_FLOAT, 0, 3);
        blurShader->enableAttributeArray("position0");

        auto curvatureFlowShader = getProgram(ParticlePrograms::CurvatureFlow);
        curvatureFlowShader->setAttributeBuffer("position0", GL_FLOAT, 0, 3);
        curvatureFlowShader->enableAttributeArray("position0");

        auto liquidShader = getProgram(ParticlePrograms::Liquid);
        liquidShader->setAttributeBuffer("position0", GL_FLOAT, 0, 3);
        liquidShader->enableAttributeArray("position0");
    }

    void ParticleShaderCollection::setBlurDirection(const QVector2D& direction) {
        auto blurShader = getProgram(ParticlePrograms::Blur);
        blurShader->bind();
        blurShader->setUniformValue(blurDirection, direction);
        blurShader->release();
    }
}
