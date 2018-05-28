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

#ifndef MODEL_FILE_HPP
#define MODEL_FILE_HPP

#include <assimp/vector3.h>
#include <assimp/matrix4x4.h>

// Forward declare assimp types
struct aiMesh;
struct aiMaterial;
struct aiNode;
struct aiScene;

namespace IO {
    class ModelFile {
    public:
        struct Material {
            QString materialName;
            QVector3D ambientLight;
            QVector3D diffuseLight;
            QVector3D specularLight;
            QVector3D emissiveLight;
            float shininess;
            bool twoSided;
        };

        struct Mesh {
            QString meshName;
            uint indexCount;    //< numFaces = indexCount / 3
            uint indexOffset;
            Material* material;
        };

        struct Node {
            QString nodeName;
            QMatrix4x4 transform;
            QVector<Mesh*> meshes;
            QVector<Node> nodes;
        };

    public:
        ModelFile(const QString& filename);
        ~ModelFile() = default;

    public:
        bool loadModel();

    public:
        const Node* getRootNode() const {
            return &rootNode;
        }

        void getBoundingBox(QVector3D* min, QVector3D* max) const;

    private:
        void loadMaterials(const aiScene*);
        void loadMeshes(const aiScene*);

        void loadIndices(const aiMesh*);
        void loadVertices(const aiMesh*);

        void loadNormals(const aiMesh*);
        void loadVertexColors(const aiMesh*);

        Material processMaterial(const aiMaterial*);
        Mesh processMesh(const aiMesh*);
        void processNodes(const aiScene*, const aiNode*, Node*);

    private:
        void getBoundingBox(const aiScene*, const aiNode*, aiVector3D* min, aiVector3D* max) const;
        void getBoundingBox(const aiScene*, aiVector3D* min, aiVector3D* max) const;
        void getAbsoluteTransform(const aiNode*, aiMatrix4x4* transform) const;

    private:
        QString fileName;

        aiVector3D bboxMinimum;
        aiVector3D bboxMaximum;

        QVector<uint> indices;
        QVector<QVector3D> vertices;
        QVector<QVector3D> normals;
        QVector<QVector4D> colors;

        Node rootNode;
        QVector<Material> materials;
        QVector<Mesh> meshes;

    public:
        DEF_GETTER(getFileName, fileName);

        DEF_GETTER(getIndices, indices);
        DEF_GETTER(getVertices, vertices);

        DEF_GETTER(getNormals, normals);
        DEF_GETTER(getColors, colors);
    };

    typedef QSharedPointer<ModelFile> ModelFilePtr;
}

Q_DECLARE_METATYPE(IO::ModelFilePtr);

#endif
