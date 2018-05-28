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
#include "ModelInstance.hpp"

namespace Scene {
    REGISTER_METATYPE(ModelInstancePtr);

    ModelInstance::ModelInstance(const IO::ModelFilePtr& file)
        : modelFile(file) {

        // TODO: Use config
        QVector3D worldSize(1.5f, 1.0f, 1.0f);

        QVector3D bboxMin, bboxMax;
        file->getBoundingBox(&bboxMin, &bboxMax);

        QVector3D bboxVec = bboxMax - bboxMin;
        scaleFactor = 0.5f * (worldSize.length() / bboxVec.length());
        updateMatrix();
    }

    void ModelInstance::moveTo(const QVector3D& pos) {
        position = pos;
        updateMatrix();
    }

    void ModelInstance::rotate(float angle, const QVector3D& axis) {
        matRotation.rotate(angle, axis);
        updateMatrix();
    }

    void ModelInstance::scale(float scale) {
        currentScale += scale;
        if (currentScale < 0.1f) {
            currentScale = 0.1f;
        }
        updateMatrix();
    }

    void ModelInstance::updateMatrix() {
        QVector3D bboxMin, bboxMax;
        modelFile->getBoundingBox(&bboxMin, &bboxMax);

        matTransform.setToIdentity();
        matTransform.translate(position);
        matTransform.scale(scaleFactor * currentScale);
        matTransform *= matRotation;

        QVector3D translation;
        translation.setY(-bboxMin.y());
        matTransform.translate(translation);

        boundingBox.setCorners(bboxMin, bboxMax);
        boundingBox.transform(matTransform);

        emit matrixChanged(matTransform);
    }

    QVector<ModelFace> ModelInstance::getFaces() const {
        QVector<ModelFace> faces;
        faces.reserve(modelFile->getIndices().count() / 3);
        getFaces(modelFile->getRootNode(), &faces);
        return faces;
    }

    void ModelInstance::getFaces(const IO::ModelFile::Node* node,
                                 QVector<ModelFace>* modelFaces) const {

        auto& indices = modelFile->getIndices();
        auto& vertices = modelFile->getVertices();

        // Transformation matrix for this node
        QMatrix4x4 faceTransform = matTransform * node->transform;

        // Walk all meshes in this node and grab their faces
        for (auto& mesh : node->meshes) {
            // Mesh consists only of triangles (divisible by 3)
            for (uint i = 0; i < mesh->indexCount / 3; ++i) {
                ModelFace modelFace;
                uint index = mesh->indexOffset + i * 3;

                for (uint j = 0; j < 3; ++j) {
                    auto& vertex = vertices[indices[index + j]];
                    modelFace.vertices[j] = faceTransform * vertex;
                }

                QVector3D u = modelFace.vertices[1] - modelFace.vertices[0];
                QVector3D v = modelFace.vertices[2] - modelFace.vertices[0];
                QVector3D n = QVector3D::crossProduct(u, v);
                modelFace.normal = n.normalized();

                modelFaces->append(std::move(modelFace));
            }
        }

        // One node can have multiple children, traverse children
        for (auto& nextNode : node->nodes) {
            getFaces(&nextNode, modelFaces);
        }
    }
}
