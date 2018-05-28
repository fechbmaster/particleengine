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

#ifndef MODEL_INSTANCE_HPP
#define MODEL_INSTANCE_HPP

#include "IO/ModelFile.hpp"
#include "BoundingBox.hpp"

namespace Scene {
    struct ModelFace {
        QVector3D vertices[3];
        QVector3D normal;
    };

    class ModelInstance : public QObject {
        Q_OBJECT

    public:
        ModelInstance(const IO::ModelFilePtr&);
        ~ModelInstance() = default;

    public:
        void moveTo(const QVector3D& position);
        void rotate(float angle, const QVector3D& axis);
        void scale(float scale);

        QVector<ModelFace> getFaces() const;

    signals:
        void matrixChanged(const QMatrix4x4&);

    private:
        void updateMatrix();
        void getFaces(const IO::ModelFile::Node*, QVector<ModelFace>*) const;

    private:
        QMatrix4x4 matTransform;
        QMatrix4x4 matRotation;

        QVector3D position;

        float scaleFactor = 1.0f;
        float currentScale = 1.0f;

        BoundingBox boundingBox;

        IO::ModelFilePtr modelFile;

    public:
        DEF_GETTER(getModelFile, modelFile);
        DEF_GETTER(getTransform, matTransform);
        DEF_GETTER(getBoundingBox, boundingBox);
    };

    typedef QSharedPointer<ModelInstance> ModelInstancePtr;
}

Q_DECLARE_METATYPE(Scene::ModelInstancePtr);

#endif
