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

#ifndef MODEL_RENDER_HPP
#define MODEL_RENDER_HPP

#include "AbstractRender.hpp"
#include "ModelInstance.hpp"
#include "IO/ModelFile.hpp"

#include "Utils/OpenGLFunctions.hpp"

#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>

namespace Scene {
    class ModelRender : public AbstractRender, private Utils::OpenGLFunctions {
        Q_OBJECT

    public:
        ModelRender(const IO::ModelFilePtr&, QObject* parent = nullptr);
        ~ModelRender() override = default;

    public:
        void initialize(AbstractScene* parent) override;
        void cleanup() override;
        void update() override;
        void render() override;

        void addRenderInstance(const ModelInstancePtr& instance);

    private:
        void initUniformLocations();
        void initVertexBuffer();
        void initIndexBuffer();
        void initNormalBuffer();
        void initColorBuffer();
        void initInstanceBuffer();

    private:
        void renderNode(const IO::ModelFile::Node* node);
        void setMaterialUniforms(const IO::ModelFile::Material*);
        void updateInstanceBuffers();

    private slots:
        void instanceChanged();

    private:
        IO::ModelFilePtr modelFile;
        QVector<ModelInstancePtr> modelInstances;

        QOpenGLBuffer vertexBuffer;
        QOpenGLBuffer normalBuffer;
        QOpenGLBuffer colorBuffer;
        QOpenGLBuffer indexBuffer;
        QOpenGLBuffer matrixBuffer;

        QOpenGLVertexArrayObject modelVAO;
        QOpenGLShaderProgram modelShader;

        int ambientLight = 0;
        int diffuseLight = 0;
        int specularLight = 0;
        int emissiveLight = 0;
        int shininess = 0;

        int nodeTransform = 0;
        int modelTransform = 0;

        bool isDirty = true;

    public:
        DEF_GETTER(getModelFile, modelFile);
    };

    typedef QSharedPointer<ModelRender> ModelRenderPtr;
}

Q_DECLARE_METATYPE(Scene::ModelRenderPtr);

#endif
