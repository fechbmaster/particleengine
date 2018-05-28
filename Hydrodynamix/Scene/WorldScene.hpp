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

#ifndef WORLD_SCENE_HPP
#define WORLD_SCENE_HPP

#include "AbstractScene.hpp"
#include "AbstractRender.hpp"
#include "PerspectiveCamera.hpp"

#include "Utils/OpenGLFunctions.hpp"
#include "Utils/UniformBufferObject.hpp"

#include <QPoint>

namespace Scene {
    class WorldScene : public AbstractScene, private Utils::OpenGLFunctions {
        Q_OBJECT

    public:
        WorldScene(QObject* parent = nullptr);
        ~WorldScene() override = default;

    public:
        void initialize() override;
        void cleanup() override;
        void update() override;
        void render() override;
        void resize(int w, int h) override;

        void addRenderer(const AbstractRenderPtr& render) override;
        void bindGlobalParams(QOpenGLShaderProgram* prog) override;

        void updateProperties(Data::SimulationConfig* config);
        void updateMousePosition(int x, int y) override;

    public:
        QVector3D projectPoint(const QVector3D&) const override;
        QVector3D unprojectPoint(const QVector3D&) const override;

    public:
        QVector3D getMousePosition() const override {
            return globalParams.mousePosition.toVector3D();
        }

        QVector4D getViewport() const override {
            return globalParams.viewport;
        }

        Camera* getActiveCamera() const override {
            return activeCamera.data();
        }

        Data::SimulationConfig* getProperties() const override {
            return simulationConfig;
        }

    private:
        void updateGlobalParams();
        void updateWorldMousePosition();

    private:
        QScopedPointer<PerspectiveCamera> activeCamera;
        QVector<AbstractRenderPtr> renderers;
        QVector<AbstractRenderPtr> newRenderers;

        struct GlobalParams {
            float matView[4][4];
            float matProj[4][4];
            QVector4D eyePosition;
            QVector4D mousePosition;
            QVector4D viewport;
        };

        QPoint lastMousePosition;

        GlobalParams globalParams;
        Utils::UniformBufferObject globalParamsBuffer;

        Data::SimulationConfig* simulationConfig = nullptr;
    };
}

#endif
