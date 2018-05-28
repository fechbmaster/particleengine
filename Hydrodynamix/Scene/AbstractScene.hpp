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

#ifndef ABSTRACT_SCENE_HPP
#define ABSTRACT_SCENE_HPP

#include "AbstractRender.hpp"
#include "Data/SimulationConfig.hpp"

#include <QOpenGLShaderProgram>

namespace Scene {
    class Camera;

    class AbstractScene : public QObject {
        Q_OBJECT

    public:
        AbstractScene(QObject* parent = nullptr)
            : QObject(parent) {}

        virtual ~AbstractScene() = default;

    public:
        virtual void initialize() = 0;
        virtual void cleanup() = 0;
        virtual void update() = 0;
        virtual void render() = 0;
        virtual void resize(int w, int h) = 0;

        virtual void addRenderer(const AbstractRenderPtr& render) = 0;
        virtual void bindGlobalParams(QOpenGLShaderProgram* prog) = 0;

        virtual void updateProperties(Data::SimulationConfig* config) = 0;
        virtual void updateMousePosition(int x, int y) = 0;

    public:
        virtual QVector3D projectPoint(const QVector3D&) const = 0;
        virtual QVector3D unprojectPoint(const QVector3D&) const = 0;

    public:
        virtual QVector3D getMousePosition() const = 0;
        virtual QVector4D getViewport() const = 0;

        virtual Camera* getActiveCamera() const = 0;
        virtual Data::SimulationConfig* getProperties() const = 0;

    signals:
        void sizeChanged(int w, int h);
    };
}

#endif
