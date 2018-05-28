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

#ifndef SIMULATION_ROOM_RENDER_HPP
#define SIMULATION_ROOM_RENDER_HPP

#include "AbstractRender.hpp"

#include "Utils/OpenGLFunctions.hpp"
#include "Data/SimulationConfig.hpp"

#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

namespace Scene {
    class SimulationRoomRender : public AbstractRender, private Utils::OpenGLFunctions {
        Q_OBJECT

    public:
        SimulationRoomRender(QObject* parent = nullptr);
        ~SimulationRoomRender() override = default;

    public:
        void initialize(AbstractScene* parent) override;
        void cleanup() override;
        void update() override;
        void render() override;

    private:
        void createGridVertices();
        void createBoxVertices();
        void createBoxOutline();

    private:
        QOpenGLShaderProgram gridShader;
        QOpenGLShaderProgram wallShader;
        QOpenGLShaderProgram outlineShader;

        QOpenGLBuffer gridVertices;
        QOpenGLVertexArrayObject gridVAO;

        QOpenGLBuffer wallVertices;
        QOpenGLVertexArrayObject wallVAO;

        QOpenGLBuffer outlineVertices;
        QOpenGLVertexArrayObject outlineVAO;

        Data::SimulationConfig* properties = nullptr;
    };

    typedef QSharedPointer<SimulationRoomRender> SimulationRoomRenderPtr;
}

Q_DECLARE_METATYPE(Scene::SimulationRoomRenderPtr);

#endif
