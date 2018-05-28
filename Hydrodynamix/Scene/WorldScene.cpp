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
#include "WorldScene.hpp"
#include "PerspectiveCamera.hpp"

#include <cstring>

#include <QVector3D>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>

namespace Scene {
    WorldScene::WorldScene(QObject* parent)
        : AbstractScene(parent)
        , activeCamera(new PerspectiveCamera) {

        memset(&globalParams, 0, sizeof(globalParams));
    }

    void WorldScene::initialize() {
        initializeOpenGLFunctions();

        glEnable(GL_DEPTH_TEST);
        glClearDepth(1.0f);

        globalParamsBuffer.create();
        globalParamsBuffer.allocate(sizeof(globalParams));

        activeCamera->setPosition(QVector3D(-1.0f, 0.8f, -1.5f));
        activeCamera->setTarget(QVector3D(0.0f, -0.2f, 0.0f));

        for (auto& renderer : renderers) {
            renderer->initialize(this);
        }
        newRenderers.clear();
    }

    void WorldScene::cleanup() {
        for (auto& renderer : renderers) {
            renderer->cleanup();
        }
    }

    void WorldScene::update() {
        updateGlobalParams();

        for (auto& renderer : newRenderers) {
            renderer->initialize(this);
        }
        newRenderers.clear();

        for (auto& renderer : renderers) {
            renderer->update();
        }
    }

    void WorldScene::render() {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        globalParamsBuffer.bind();
        for (auto& renderer : renderers) {
            renderer->render();
        }
        globalParamsBuffer.release();
    }

    void WorldScene::resize(int w, int h) {
        activeCamera->setAspect((float) w / (float) h);
        emit sizeChanged(w, h);
    }

    void WorldScene::addRenderer(const AbstractRenderPtr& renderer) {
        renderers.append(renderer);
        newRenderers.append(renderer);
    }

    void WorldScene::bindGlobalParams(QOpenGLShaderProgram* prog) {
        globalParamsBuffer.bind(prog->programId(), "GlobalParams");
    }

    void WorldScene::updateMousePosition(int x, int y) {
        lastMousePosition = QPoint(x, y);
    }
   
    void WorldScene::updateProperties(Data::SimulationConfig* config) {
        simulationConfig = config;
    }

    void WorldScene::updateGlobalParams() {
        auto& matView = activeCamera->getViewMatrix();
        auto& matProj = activeCamera->getProjMatrix();

        matView.copyDataTo((float*) &globalParams.matView);
        matProj.copyDataTo((float*) &globalParams.matProj);

        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        globalParams.viewport = QVector4D(vp[0], vp[1], vp[2], vp[3]);
        globalParams.eyePosition = activeCamera->getPosition();

        updateWorldMousePosition();
        globalParamsBuffer.write(0, &globalParams, sizeof(globalParams));
    }

    // Calculate the point of intersection where our cursor position hits the floor (in world space).
    void WorldScene::updateWorldMousePosition() {
        auto generalProperties = simulationConfig->getGeneralProperties();
        float size = generalProperties.worldSizeY / 2.0f;

        QVector3D planeNormal = { 0.0f, +1.0f, 0.0f };
        QVector3D planeOffset = { 0.0f, -size, 0.0f };

        float mouseX = (float) lastMousePosition.x();
        float mouseY = (float) lastMousePosition.y();

        QVector3D rayNear = unprojectPoint(QVector3D(mouseX, mouseY, 0.0f));
        QVector3D rayFar = unprojectPoint(QVector3D(mouseX, mouseY, 1.0f));

        float planeDot = QVector3D::dotProduct(planeNormal, planeOffset - rayNear);
        float rayDot = QVector3D::dotProduct(planeNormal, rayFar - rayNear);
        float distance = planeDot / rayDot;

        if (distance >= 0.0f && distance <= 1.0f) {
            globalParams.mousePosition = rayNear + (rayFar - rayNear) * distance;
        }
    }

    // Transform world coordinates into 2D screen coords (the Z coord tells how far from the camera it is).
    // If the Z coord is <= 0, the point is not visible on the screen.
    QVector3D WorldScene::projectPoint(const QVector3D& world) const {
        QVector3D position = world;
        position = activeCamera->getViewMatrix() * position;
        position = activeCamera->getProjMatrix() * position;

        float vpX = globalParams.viewport.x();
        float vpY = globalParams.viewport.y();

        float vpWidth = globalParams.viewport.z();
        float vpHeight = globalParams.viewport.w();

        float minDepth = 0.0f;
        float maxDepth = 1.0f;

        position.setX(vpX + (1.0f + position.x()) * vpWidth / 2.0f);
        position.setY(vpY + (1.0f - position.y()) * vpHeight / 2.0f);
        position.setZ(minDepth + position.z() * (maxDepth - minDepth));
        return position;
    }

    // Does the exact opposite of projectPoint (screen -> world coords).
    QVector3D WorldScene::unprojectPoint(const QVector3D& screen) const {
        float vpX = globalParams.viewport.x();
        float vpY = globalParams.viewport.y();

        float vpWidth = globalParams.viewport.z();
        float vpHeight = globalParams.viewport.w();

        float minDepth = 0.0f;
        float maxDepth = 1.0f;

        QVector3D position = screen;
        position.setX(2.0f * (position.x() - vpX) / vpWidth - 1.0f);
        position.setY(1.0f - 2.0f * (position.y() - vpY) / vpHeight);
        position.setZ((position.z() - minDepth) / (maxDepth - minDepth));

        position = activeCamera->getProjInverse() * position;
        position = activeCamera->getViewInverse() * position;
        return position;
    }
}
