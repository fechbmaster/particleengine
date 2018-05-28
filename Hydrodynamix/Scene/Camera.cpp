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
#include "Camera.hpp"

namespace Scene {
    Camera::Camera() {
        position = { 0, 0, 0 };
        forward = { 1, 0, 0 };
        up = { 0, 1, 0 };
        right = { 0, 0, 1 };

        updateView();
    }

    void Camera::setPosition(const QVector3D& pos) {
        position = pos;
        updateView();
    }

    void Camera::setTarget(const QVector3D& tar) {
        forward = tar - position;
        forward.normalize();

        right = QVector3D::crossProduct(forward, QVector3D(0, 1, 0));
        right.normalize();

        if (right.lengthSquared() < 0.5f) {
            right = QVector3D(0, 0, 1);
        }

        up = QVector3D::crossProduct(right, forward);
        updateView();
    }

    void Camera::moveUp(float amount) {
        move(QVector3D(0, 1, 0) * amount);
    }

    void Camera::moveForward(float amount) {
        move(forward * amount);
    }

    void Camera::moveRight(float amount) {
        move(right * amount);
    }

    void Camera::move(const QVector3D& direction) {
        position += direction;
        updateView();
    }

    void Camera::pitch(float angle) {
        QMatrix4x4 matRot;
        matRot.rotate(angle, right);

        up = matRot * up;
        if (up.y() < 0) {
            up.setY(0);
        }

        up.normalize();

        forward = QVector3D::crossProduct(up, right);
        updateView();
    }

    void Camera::yaw(float angle) {
        QMatrix4x4 matRot;
        matRot.rotate(angle, QVector3D(0, 1, 0));

        forward = matRot * forward;
        forward.normalize();

        right = matRot * right;
        right.normalize();

        up = QVector3D::crossProduct(right, forward);
        updateView();
    }

    void Camera::roll(float angle) {
        QMatrix4x4 matRot;
        matRot.rotate(angle, forward);

        up = matRot * up;
        if (up.y() < 0) {
            up.setY(0);
        }

        up.normalize();

        right = QVector3D::crossProduct(forward, up);
        updateView();
    }

    void Camera::updateView() {
        matView.setToIdentity();
        matView.lookAt(position, position + forward, up);
        matViewInverse = matView.inverted();
    }
}
