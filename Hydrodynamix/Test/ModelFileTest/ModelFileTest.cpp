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

#include "../../IO/ModelFile.hpp"

#include <QTest>

class ModelFileTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void testLoadModel();
    void testVertices();
    void testNormals();
    void testIndices();
    void testBoundingBox();
    void cleanupTestCase();

private:
    IO::ModelFilePtr modelFile;
};

void ModelFileTest::initTestCase() {
    modelFile.reset(new IO::ModelFile("Test.obj"));
}

void ModelFileTest::testLoadModel() {
    QVERIFY(modelFile->loadModel());
}

void ModelFileTest::testVertices() {
    auto vertices = modelFile->getVertices();
    QVERIFY(vertices.count() == 8);

    QVector<QVector3D> fileVertices;
    fileVertices.append({ 1.000000f, 0.000000f, -1.000000f });
    fileVertices.append({ 1.00000f, 0.000000f, 1.000000f });
    fileVertices.append({ -1.000000f, 0.000000f, 1.000000f });
    fileVertices.append({ -1.000000f, 0.000000f, -1.000000f });
    fileVertices.append({ 1.000000f, 2.000000f, -1.000000f });
    fileVertices.append({ 1.000000f, 2.000000f, 1.000000f });
    fileVertices.append({ -1.000000f, 2.000000f, 1.000000f });
    fileVertices.append({ -1.000000f, 2.000000f, -1.000000f });

    for (auto& vertex : vertices) {
        fileVertices.removeAt(fileVertices.indexOf(vertex));
    }
    QVERIFY(fileVertices.isEmpty());
    fileVertices.clear();
}

void ModelFileTest::testNormals() {
    auto normals = modelFile->getNormals();
    auto vertices = modelFile->getVertices();

    QVector3D qVec1, qVec2;
    modelFile->getBoundingBox(&qVec1, &qVec2);

    qVec1 += qVec2;
    QVector3D center = qVec1 * 0.5f;

    QVERIFY(normals.count() == 8);
    QVERIFY(vertices.count() == 8);
    for (int i = 0; i < normals.count(); i++) {
        QVERIFY(normals.at(i).length() < 1.01);
        QVERIFY(normals.at(i).length() > 0.99);
        float vertexToCenter = vertices.at(i).distanceToPoint(center);
        float normalToCenter = (vertices.at(i) + normals.at(i)).distanceToPoint(center);
        QVERIFY(vertexToCenter + 0.9 < normalToCenter);
    }
}

void ModelFileTest::testIndices() {
    auto vertices = modelFile->getVertices();
    auto indices = modelFile->getIndices();
    QVERIFY(indices.count() == 6 * 2 * 3);
    for (int index : indices) {
        QVERIFY(index >= 0);
        QVERIFY(index < 8);
    }

    for (int i = 2; i < indices.count(); i+=3) {
        QVector3D v1 = vertices.at(indices.at(i-2));
        QVector3D v2 = vertices.at(indices.at(i-1));
        QVector3D v3 = vertices.at(indices.at(i));
        QVERIFY(v1 != v2);
        QVERIFY(v1 != v3);
        QVERIFY(v2 != v3);
        bool x12 = v1.x() == v2.x();
        bool y12 = v1.y() == v2.y();
        bool z12 = v1.z() == v2.z();
        bool x13 = v1.x() == v3.x();
        bool y13 = v1.y() == v3.y();
        bool z13 = v1.z() == v3.z();
        bool x23 = v2.x() == v3.x();
        bool y23 = v2.y() == v3.y();
        bool z23 = v2.z() == v3.z();
        bool is12cathetus = (x12 && y12) || (x12 && z12) || (y12 && z12);
        bool is13cathetus = (x13 && y13) || (x13 && z13) || (y13 && z13);
        bool is23cathetus = (x23 && y23) || (x23 && z23) || (y23 && z23);
        QVERIFY((is12cathetus && is13cathetus && !is23cathetus) || (is12cathetus && !is13cathetus && is23cathetus) || (!is12cathetus && is13cathetus && is23cathetus));
        QVERIFY((x12 && x13 && x23) || (y12 && y13 && y23) || (z12 && z13 && z23));
    }
}

void ModelFileTest::testBoundingBox() {
    QVector3D min, max;
    modelFile->getBoundingBox(&min, &max);

    QVERIFY(min.x() == -1);
    QVERIFY(min.y() == 0);
    QVERIFY(min.z() == -1);
    QVERIFY(max.x() == 1);
    QVERIFY(max.y() == 2);
    QVERIFY(max.z() == 1);
}

void ModelFileTest::cleanupTestCase() {
    modelFile.clear();
}

QTEST_MAIN(ModelFileTest)
#include "modelfiletest.moc"
