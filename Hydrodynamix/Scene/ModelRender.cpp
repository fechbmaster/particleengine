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
#include "ModelRender.hpp"
#include "AbstractScene.hpp"

namespace Scene {
    REGISTER_METATYPE(ModelRenderPtr);

    ModelRender::ModelRender(const IO::ModelFilePtr& file, QObject* parent)
        : AbstractRender(parent)
        , modelFile(file)
        , indexBuffer(QOpenGLBuffer::IndexBuffer) {

    }

    void ModelRender::initialize(AbstractScene* parent) {
        initializeOpenGLFunctions();

        modelShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "Shader/ModelVertex.glsl");
        modelShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "Shader/ModelFragment.glsl");
        modelShader.link();

        parent->bindGlobalParams(&modelShader);

        modelVAO.create();
        QOpenGLVertexArrayObject::Binder binder(&modelVAO);

        initUniformLocations();
        initVertexBuffer();
        initIndexBuffer();
        initNormalBuffer();
        initColorBuffer();
        initInstanceBuffer();
    }

    void ModelRender::initUniformLocations() {
        ambientLight = modelShader.uniformLocation("ambientLight");
        diffuseLight = modelShader.uniformLocation("diffuseLight");
        specularLight = modelShader.uniformLocation("specularLight");
        emissiveLight = modelShader.uniformLocation("emissiveLight");
        shininess = modelShader.uniformLocation("shininess");

        modelTransform = modelShader.uniformLocation("modelTransform");
        nodeTransform = modelShader.uniformLocation("nodeTransform");
    }

    void ModelRender::initVertexBuffer() {
        vertexBuffer.create();
        vertexBuffer.bind();
        vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);

        auto& vertices = modelFile->getVertices();
        vertexBuffer.allocate(vertices.data(), vertices.count() * sizeof(QVector3D));

        modelShader.setAttributeBuffer("position0", GL_FLOAT, 0, 3);
        modelShader.enableAttributeArray("position0");
    }

    void ModelRender::initIndexBuffer() {
        indexBuffer.create();
        indexBuffer.bind();
        indexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);

        auto& indices = modelFile->getIndices();
        indexBuffer.allocate(indices.data(), indices.count() * sizeof(uint));
    }

    void ModelRender::initNormalBuffer() {
        normalBuffer.create();
        normalBuffer.bind();
        normalBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);

        auto& normals = modelFile->getNormals();
        normalBuffer.allocate(normals.data(), normals.count() * sizeof(QVector3D));

        modelShader.setAttributeBuffer("normal0", GL_FLOAT, 0, 3);
        modelShader.enableAttributeArray("normal0");
    }

    void ModelRender::initColorBuffer() {
        colorBuffer.create();
        colorBuffer.bind();
        colorBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);

        auto& colors = modelFile->getColors();
        colorBuffer.allocate(colors.data(), colors.count() * sizeof(QVector4D));

        modelShader.setAttributeBuffer("color0", GL_FLOAT, 0, 4);
        modelShader.enableAttributeArray("color0");
    }

    void ModelRender::initInstanceBuffer() {
        matrixBuffer.create();
        matrixBuffer.bind();
        matrixBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);

        int transform0 = modelShader.attributeLocation("transform0");
        for (int i = 0; i < 4; ++i) {
            modelShader.setAttributeBuffer(transform0 + i, GL_FLOAT,
                sizeof(QVector4D) * i, 4, sizeof(QVector4D) * 4);

            modelShader.enableAttributeArray(transform0 + i);
            glVertexAttribDivisor(transform0 + i, 1);
        }
    }

    void ModelRender::cleanup() {

    }

    void ModelRender::update() {
        if (isDirty) {
            updateInstanceBuffers();
            isDirty = false;
        }
    }

    void ModelRender::addRenderInstance(const ModelInstancePtr& instance) {
        Q_ASSERT(instance->getModelFile() == modelFile);

        isDirty = true;
        modelInstances.append(instance);
        connect(instance.data(), &ModelInstance::matrixChanged, this, &ModelRender::instanceChanged);
    }

    void ModelRender::renderNode(const IO::ModelFile::Node* node) {
        // Each node also has a transformation matrix
        modelShader.setUniformValue(nodeTransform, node->transform);

        for (auto& mesh : node->meshes) {
            setMaterialUniforms(mesh->material);

            glDrawElementsInstanced(GL_TRIANGLES, mesh->indexCount, GL_UNSIGNED_INT,
                (GLvoid*) (mesh->indexOffset * sizeof(uint)), modelInstances.size());
        }

        for (auto& child : node->nodes) {
            renderNode(&child);
        }
    }

    void ModelRender::setMaterialUniforms(const IO::ModelFile::Material* material) {
        modelShader.setUniformValue(ambientLight, material->ambientLight);
        modelShader.setUniformValue(diffuseLight, material->diffuseLight);
        modelShader.setUniformValue(specularLight, material->specularLight);
        modelShader.setUniformValue(emissiveLight, material->emissiveLight);
        modelShader.setUniformValue(shininess, material->shininess);

        if (material->twoSided) {
            glDisable(GL_CULL_FACE);
        } else {
            glEnable(GL_CULL_FACE);
        }
    }

    void ModelRender::updateInstanceBuffers() {
        QVector<float> matrices;
        matrices.reserve(modelInstances.size() * 16);

        for (auto& instance : modelInstances) {
            QMatrix4x4 transform = instance->getTransform();
            const float* matrixData = transform.constData();

            for (int i = 0; i < 16; ++i) {
                matrices.append(matrixData[i]);
            }
        }

        matrixBuffer.bind();
        int bufferSize = matrices.size() * sizeof(float);
        matrixBuffer.allocate(matrices.data(), bufferSize);
    }

    void ModelRender::instanceChanged() {
        isDirty = true;
    }

    void ModelRender::render() {
        modelShader.bind();
        {
            QOpenGLVertexArrayObject::Binder binder(&modelVAO);
            renderNode(modelFile->getRootNode());
        }
        modelShader.release();
    }
}
