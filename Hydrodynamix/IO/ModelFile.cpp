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
#include "ModelFile.hpp"

#include <limits>
#include <algorithm>

#include <assimp/Importer.hpp>
#include <assimp/vector3.h>
#include <assimp/matrix4x4.h>
#include <assimp/color4.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>

#pragma comment(lib, "assimp.lib")

namespace IO {
    REGISTER_METATYPE(ModelFilePtr);

    static QVector3D aiVectorToQVector(const aiVector3D& vec) {
        return QVector3D(vec.x, vec.y, vec.z);
    }

    static QVector4D aiColorToQVector(const aiColor4D& color) {
        return QVector4D(color.r, color.g, color.b, color.a);
    }

    ModelFile::ModelFile(const QString& filename)
        : fileName(filename) {
    }

    bool ModelFile::loadModel() {
        Assimp::Importer importer;
        if (!importer.ReadFile(fileName.toStdString(), aiProcessPreset_TargetRealtime_MaxQuality)) {
            qDebug() << "Error loading file (assimp):" << importer.GetErrorString();
            return false;
        }

        const aiScene* scene = importer.GetScene();
        getBoundingBox(scene, &bboxMinimum, &bboxMaximum);

        loadMaterials(scene);
        loadMeshes(scene);

        if (scene->mRootNode) {
            processNodes(scene, scene->mRootNode, &rootNode);
        }
        return true;
    }

    void ModelFile::loadMaterials(const aiScene* scene) {
        if (scene->HasMaterials()) {
            materials.resize(scene->mNumMaterials);
            for (uint i = 0; i < scene->mNumMaterials; ++i) {
                materials[i] = processMaterial(scene->mMaterials[i]);
            }
        }
    }

    void ModelFile::loadMeshes(const aiScene* scene) {
        if (scene->HasMeshes()) {
            meshes.resize(scene->mNumMeshes);
            for (uint i = 0; i < scene->mNumMeshes; ++i) {
                meshes[i] = processMesh(scene->mMeshes[i]);
            }
        }
    }

    ModelFile::Material ModelFile::processMaterial(const aiMaterial* material) {
        Material meshMaterial;

        aiString materialName;
        material->Get(AI_MATKEY_NAME, materialName);
        if (materialName.length > 0) {
            meshMaterial.materialName = materialName.C_Str();
        }

        aiColor3D ambient, diffuse, specular, emissive;
        material->Get(AI_MATKEY_COLOR_AMBIENT, ambient);
        material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
        material->Get(AI_MATKEY_COLOR_SPECULAR, specular);
        material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);

        float shininess = 0.0f;
        material->Get(AI_MATKEY_SHININESS, shininess);
        meshMaterial.shininess = shininess;

        int twoSided = 0;
        material->Get(AI_MATKEY_TWOSIDED, twoSided);
        meshMaterial.twoSided = (twoSided != 0);

        meshMaterial.ambientLight = QVector3D(ambient.r, ambient.g, ambient.b);
        meshMaterial.diffuseLight = QVector3D(diffuse.r, diffuse.g, diffuse.b);
        meshMaterial.specularLight = QVector3D(specular.r, specular.g, specular.b);
        meshMaterial.emissiveLight = QVector3D(emissive.r, emissive.g, emissive.b);
        return meshMaterial;
    }

    void ModelFile::loadVertices(const aiMesh* mesh) {
        uint oldVertexCount = vertices.count();
        vertices.reserve(oldVertexCount + mesh->mNumVertices);

        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            aiVector3D vertex = mesh->mVertices[i];
            vertices.append(aiVectorToQVector(vertex));
        }
    }

    void ModelFile::loadIndices(const aiMesh* mesh) {
        uint vertexOffset = vertices.size();
        indices.reserve(indices.count() + mesh->mNumFaces * 3);

        for (uint i = 0; i < mesh->mNumFaces; ++i) {
            const aiFace* face = &mesh->mFaces[i];
            if (face->mNumIndices != 3) {
                qDebug() << "Warning: Mesh face with" << face->mNumIndices << "indices, ignoring this primitive.";
                continue;
            }

            indices.append(face->mIndices[0] + vertexOffset);
            indices.append(face->mIndices[1] + vertexOffset);
            indices.append(face->mIndices[2] + vertexOffset);
        }
    }

    void ModelFile::loadNormals(const aiMesh* mesh) {
        uint oldNormalCount = normals.count();
        normals.reserve(oldNormalCount + mesh->mNumVertices);

        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            aiVector3D normal = { 0, 1, 0 };
            if (mesh->HasNormals()) {
                normal = mesh->mNormals[i];
            }
            normals.append(aiVectorToQVector(normal));
        }
    }

    void ModelFile::loadVertexColors(const aiMesh* mesh) {
        uint oldColorCount = colors.count();
        colors.reserve(oldColorCount + mesh->mNumVertices);

        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            aiColor4D vertexColor = { 1, 1, 1, 1 };
            if (mesh->HasVertexColors(0)) {
                vertexColor = mesh->mColors[0][i];
            }
            colors.append(aiColorToQVector(vertexColor));
        }
    }

    ModelFile::Mesh ModelFile::processMesh(const aiMesh* mesh) {
        Mesh modelMesh;
        if (mesh->mName.length > 0) {
            modelMesh.meshName = mesh->mName.C_Str();
        }

        modelMesh.indexOffset = indices.size();
        uint oldIndexCount = indices.count();

        loadIndices(mesh);
        loadVertices(mesh);
        loadNormals(mesh);
        loadVertexColors(mesh);

        modelMesh.indexCount = indices.count() - oldIndexCount;
        modelMesh.material = &materials[mesh->mMaterialIndex];
        return modelMesh;
    }

    void ModelFile::processNodes(const aiScene* scene, const aiNode* node, Node* newNode) {
        if (node->mName.length > 0) {
            newNode->nodeName = node->mName.C_Str();
        }

        aiMatrix4x4 transform;
        getAbsoluteTransform(node, &transform);
        newNode->transform = QMatrix4x4(transform[0]);

        newNode->meshes.resize(node->mNumMeshes);
        for (uint i = 0; i < node->mNumMeshes; ++i) {
            newNode->meshes[i] = &meshes[node->mMeshes[i]];
        }

        newNode->nodes.resize(node->mNumChildren);
        for (uint i = 0; i < node->mNumChildren; ++i) {
            processNodes(scene, node->mChildren[i], &newNode->nodes[i]);
        }
    }

    void ModelFile::getBoundingBox(const aiScene* scene,
                                   const aiNode* node,
                                   aiVector3D* minimum,
                                   aiVector3D* maximum) const {

        aiMatrix4x4 transform;
        getAbsoluteTransform(node, &transform);

        for (uint i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

            for (uint j = 0; j < mesh->mNumVertices; ++j) {
                aiVector3D tmp = transform * mesh->mVertices[j];

                minimum->x = std::min(minimum->x, tmp.x);
                minimum->y = std::min(minimum->y, tmp.y);
                minimum->z = std::min(minimum->z, tmp.z);

                maximum->x = std::max(maximum->x, tmp.x);
                maximum->y = std::max(maximum->y, tmp.y);
                maximum->z = std::max(maximum->z, tmp.z);
            }
        }

        for (uint i = 0; i < node->mNumChildren; ++i) {
            getBoundingBox(scene, node->mChildren[i], minimum, maximum);
        }
    }

    void ModelFile::getBoundingBox(const aiScene* scene, aiVector3D* minimum, aiVector3D* maximum) const {
        float min = std::numeric_limits<float>::min();
        float max = std::numeric_limits<float>::max();

        *minimum = aiVector3D(max, max, max);
        *maximum = aiVector3D(min, min, min);
        getBoundingBox(scene, scene->mRootNode, minimum, maximum);
    }

    void ModelFile::getAbsoluteTransform(const aiNode* node, aiMatrix4x4* transform) const {
        if (node->mParent) {
            // Recursively apply parents transformation
            getAbsoluteTransform(node->mParent, transform);
        }

        // Apply parent transformation, then child
        *transform *= node->mTransformation;
    }

    void ModelFile::getBoundingBox(QVector3D* min, QVector3D* max) const {
        *min = aiVectorToQVector(bboxMinimum);
        *max = aiVectorToQVector(bboxMaximum);
    }
}
