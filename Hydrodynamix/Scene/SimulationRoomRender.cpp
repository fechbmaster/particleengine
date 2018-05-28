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
#include "SimulationRoomRender.hpp"
#include "AbstractScene.hpp"

#include "Data/SimulationConfig.hpp"

struct BoxVertex {
    QVector3D position0;
    QVector3D normal0;
};

struct OutlineVertex {
    QVector3D position0;
};

struct GridVertex {
    QVector3D position0;
};

namespace Scene {
    REGISTER_METATYPE(SimulationRoomRenderPtr);

    SimulationRoomRender::SimulationRoomRender(QObject* parent)
        : AbstractRender(parent) {

    }

    void SimulationRoomRender::initialize(AbstractScene* parent) {
        properties = parent->getProperties();
        initializeOpenGLFunctions();

        gridShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/RoomGridVertex.glsl");
        gridShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/RoomGridFragment.glsl");
        gridShader.link();

        parent->bindGlobalParams(&gridShader);

        wallShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/RoomWallVertex.glsl");
        wallShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/RoomWallFragment.glsl");
        wallShader.link();

        parent->bindGlobalParams(&wallShader);

        outlineShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/RoomOutlineVertex.glsl");
        outlineShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/RoomOutlineFragment.glsl");
        outlineShader.link();

        parent->bindGlobalParams(&outlineShader);

        createGridVertices();
        createBoxVertices();
        createBoxOutline();
    }

    void SimulationRoomRender::createGridVertices() {
        auto generalProps = properties->getGeneralProperties();

        float gridSizeX = generalProps.worldSizeX * 2.0f;
        float gridSizeY = generalProps.worldSizeY / 2.0f;
        float gridSizeZ = generalProps.worldSizeZ * 2.0f;

        const GridVertex vertices[] = {
            { { -gridSizeX, -gridSizeY, -gridSizeZ } },
            { { -gridSizeX, -gridSizeY, +gridSizeZ } },
            { { +gridSizeX, -gridSizeY, -gridSizeZ } },
            { { +gridSizeX, -gridSizeY, +gridSizeZ } },
        };

        gridVAO.create();
        QOpenGLVertexArrayObject::Binder binder(&gridVAO);

        gridVertices.create();
        gridVertices.bind();
        gridVertices.setUsagePattern(QOpenGLBuffer::StaticDraw);
        gridVertices.allocate(vertices, sizeof(vertices));

        gridShader.setAttributeBuffer("position0", GL_FLOAT,
            offsetof(GridVertex, position0), 3, sizeof(GridVertex));

        gridShader.enableAttributeArray("position0");
    }

    void SimulationRoomRender::createBoxVertices() {
        auto generalProps = properties->getGeneralProperties();

        float boxSizeX = generalProps.worldSizeX / 2.0f;
        float boxSizeY = generalProps.worldSizeY / 2.0f;
        float boxSizeZ = generalProps.worldSizeZ / 2.0f;

        const BoxVertex vertices[] = {
            { { -boxSizeX, +boxSizeY, +boxSizeZ }, { +1.0f, +1.0f, +1.0f } },
            { { -boxSizeX, +boxSizeY, -boxSizeZ }, { +1.0f, +1.0f, -1.0f } },
            { { -boxSizeX, -boxSizeY, +boxSizeZ }, { +1.0f, +1.0f, +1.0f } },
            { { -boxSizeX, -boxSizeY, -boxSizeZ }, { +1.0f, +1.0f, -1.0f } },
            { { +boxSizeX, -boxSizeY, +boxSizeZ }, { -1.0f, +1.0f, +1.0f } },
            { { +boxSizeX, -boxSizeY, -boxSizeZ }, { -1.0f, +1.0f, -1.0f } },
            { { +boxSizeX, +boxSizeY, +boxSizeZ }, { -1.0f, +1.0f, +1.0f } },
            { { +boxSizeX, +boxSizeY, -boxSizeZ }, { -1.0f, +1.0f, -1.0f } },
        };

        wallVAO.create();
        QOpenGLVertexArrayObject::Binder binder(&wallVAO);

        wallVertices.create();
        wallVertices.bind();
        wallVertices.setUsagePattern(QOpenGLBuffer::StaticDraw);
        wallVertices.allocate(vertices, sizeof(vertices));

        wallShader.setAttributeBuffer("position0", GL_FLOAT,
            offsetof(BoxVertex, position0), 3, sizeof(BoxVertex));

        wallShader.setAttributeBuffer("normal0", GL_FLOAT,
            offsetof(BoxVertex, normal0), 3, sizeof(BoxVertex));

        wallShader.enableAttributeArray("position0");
        wallShader.enableAttributeArray("normal0");
    }

    void SimulationRoomRender::createBoxOutline() {
        auto generalProps = properties->getGeneralProperties();

        float outlineSizeX = generalProps.worldSizeX / 2.0f;
        float outlineSizeY = generalProps.worldSizeY / 2.0f;
        float outlineSizeZ = generalProps.worldSizeZ / 2.0f;

        const OutlineVertex vertices[] = {
            { { -outlineSizeX, -outlineSizeY, -outlineSizeZ } },
            { { -outlineSizeX, -outlineSizeY, +outlineSizeZ } },
            { { -outlineSizeX, +outlineSizeY, +outlineSizeZ } },
            { { -outlineSizeX, +outlineSizeY, -outlineSizeZ } },
            { { -outlineSizeX, -outlineSizeY, -outlineSizeZ } },
            { { +outlineSizeX, -outlineSizeY, -outlineSizeZ } },
            { { +outlineSizeX, -outlineSizeY, +outlineSizeZ } },
            { { -outlineSizeX, -outlineSizeY, +outlineSizeZ } },
            { { +outlineSizeX, -outlineSizeY, +outlineSizeZ } },
            { { +outlineSizeX, +outlineSizeY, +outlineSizeZ } },
            { { +outlineSizeX, +outlineSizeY, -outlineSizeZ } },
            { { +outlineSizeX, -outlineSizeY, -outlineSizeZ } },
        };

        outlineVAO.create();
        QOpenGLVertexArrayObject::Binder binder(&outlineVAO);

        outlineVertices.create();
        outlineVertices.bind();
        outlineVertices.setUsagePattern(QOpenGLBuffer::StaticDraw);
        outlineVertices.allocate(vertices, sizeof(vertices));

        outlineShader.setAttributeBuffer("position0", GL_FLOAT,
            offsetof(OutlineVertex, position0), 3, sizeof(OutlineVertex));

        outlineShader.enableAttributeArray("position0");
    }

    void SimulationRoomRender::cleanup() {

    }

    void SimulationRoomRender::update() {

    }

    void SimulationRoomRender::render() {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);

        gridShader.bind();
        {
            QOpenGLVertexArrayObject::Binder binder(&gridVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        gridShader.release();

        wallShader.bind();
        {
            QOpenGLVertexArrayObject::Binder binder(&wallVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 8);
        }
        wallShader.release();

        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);

        outlineShader.bind();
        {
            QOpenGLVertexArrayObject::Binder binder(&outlineVAO);
            glDrawArrays(GL_LINE_STRIP, 0, 12);
        }
        outlineShader.release();
    }
}
