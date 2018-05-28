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
#include "BoundingBox.hpp"

#include <limits>
#include <algorithm>

namespace Scene {
    void BoundingBox::transform(const QMatrix4x4& mat) {
        float min = std::numeric_limits<float>::min();
        float max = std::numeric_limits<float>::max();

        QVector3D newMin = { max, max, max };
        QVector3D newMax = { min, min, min };

        for (QVector3D& v : corners) {
            v = mat * v;

            newMin.setX(std::min(newMin.x(), v.x()));
            newMin.setY(std::min(newMin.y(), v.y()));
            newMin.setZ(std::min(newMin.z(), v.z()));

            newMax.setX(std::max(newMax.x(), v.x()));
            newMax.setY(std::max(newMax.y(), v.y()));
            newMax.setZ(std::max(newMax.z(), v.z()));
        }

        setCorners(newMin, newMax);
    }

    void BoundingBox::setCorners(const QVector3D& min, const QVector3D& max) {
        corners[0] = { min.x(), min.y(), min.z() };
        corners[1] = { max.x(), min.y(), min.z() };
        corners[2] = { max.x(), max.y(), min.z() };
        corners[3] = { min.x(), max.y(), min.z() };
        corners[4] = { min.x(), min.y(), max.z() };
        corners[5] = { max.x(), min.y(), max.z() };
        corners[6] = { max.x(), max.y(), max.z() };
        corners[7] = { min.x(), max.y(), max.z() };
    }

    void BoundingBox::getCorners(QVector3D corner[8]) const {
        memcpy(corner, corners, sizeof(corners));
    }
}
