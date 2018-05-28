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

#ifndef CAMERA_HPP
#define CAMERA_HPP

namespace Scene {
    class Camera {
    public:
        Camera();
        virtual ~Camera() = default;

    public:
        virtual void moveForward(float amount);
        virtual void moveRight(float amount);
        virtual void moveUp(float amount);

        virtual void roll(float angle);
        virtual void yaw(float angle);
        virtual void pitch(float angle);

        virtual void move(const QVector3D& direction);
        virtual void setPosition(const QVector3D& pos);
        virtual void setTarget(const QVector3D& target);

    private:
        void updateView();

    private:
        QMatrix4x4 matView;
        QMatrix4x4 matViewInverse;

        QVector3D position;
        QVector3D forward;
        QVector3D up;
        QVector3D right;

    public:
        DEF_GETTER(getViewMatrix, matView);
        DEF_GETTER(getViewInverse, matViewInverse);

        DEF_GETTER(getPosition, position);
        DEF_GETTER(getForward, forward);
        DEF_GETTER(getUp, up);
        DEF_GETTER(getRight, right);
    };
}

#endif
