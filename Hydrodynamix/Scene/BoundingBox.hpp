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

#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP

#include <QVector3D>
#include <QMatrix4x4>

namespace Scene {
    class BoundingBox {
    public:
        BoundingBox() = default;
        ~BoundingBox() = default;

        BoundingBox(const QVector3D& min, const QVector3D& max) {
            setCorners(min, max);
        }

    public:
        void transform(const QMatrix4x4& mat);

        void setCorners(const QVector3D& min, const QVector3D& max);
        void getCorners(QVector3D corners[8]) const;

    public:
        const QVector3D& minimum() const {
            return corners[0];
        }

        const QVector3D& maximum() const {
            return corners[6];
        }

        const QVector3D* getCorners() const {
            return corners;
        }

    private:
        QVector3D corners[8];
    };

    inline BoundingBox operator * (const QMatrix4x4& mat, const BoundingBox& bb) {
        BoundingBox bbox = bb;
        bbox.transform(mat);
        return bbox;
    }
}

#endif
