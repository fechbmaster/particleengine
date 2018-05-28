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
#include "CameraControl.hpp"
#include "InputManager.hpp"
#include "Scene/Camera.hpp"

#include <QDateTime>

namespace IO {
    void CameraControl::update() {
        qint64 now = QDateTime::currentMSecsSinceEpoch();
        float diff = (float)(now - lastUpdate) / 1000.0f;
        lastUpdate = now;

        float factor = 3.0f;
        if (movingForward) {
            activeCamera->moveForward(diff * factor * movingForward);
        }
        if (movingRight) {
            activeCamera->moveRight(diff * factor * movingRight);
        }
        if (movingUp) {
            activeCamera->moveUp(diff * factor * movingUp);
        }
    }

    void CameraControl::setTarget(Scene::Camera* camera) {
        activeCamera = camera;
    }

    void CameraControl::setInputManager(InputManager* inputManager) {
        connect(inputManager, &InputManager::mouseMovedEvent, this, &CameraControl::handleMouseMoved);
        connect(inputManager, &InputManager::mousePressedEvent, this, &CameraControl::handleMousePressed);
        connect(inputManager, &InputManager::mouseReleasedEvent, this, &CameraControl::handleMouseReleased);
        connect(inputManager, &InputManager::keyPressedEvent, this, &CameraControl::handleKeyPressed);
        connect(inputManager, &InputManager::keyReleasedEvent, this, &CameraControl::handleKeyReleased);
    }

    void CameraControl::handleMouseMoved(QMouseEvent* evt) {
        if (evt->buttons() & Qt::RightButton) {
            int dx = evt->pos().x() - lastCursorPos.x();
            int dy = evt->pos().y() - lastCursorPos.y();

            if (dx != 0) {
                activeCamera->yaw((-dx * 0.002f * 180.0f) / 3.1415926f);
            }

            if (dy != 0) {
                activeCamera->pitch((-dy * 0.002f * 180.0f) / 3.1415926f);
            }
        }

        lastCursorPos = evt->pos();
    }

    void CameraControl::handleMousePressed(QMouseEvent* evt) {
        lastCursorPos = evt->pos();
    }

    void CameraControl::handleMouseReleased(QMouseEvent* evt) {
        lastCursorPos = evt->pos();
    }

    void CameraControl::handleKeyPressed(QKeyEvent* evt) {
        switch (evt->key()) {
        case Qt::Key_W: movingForward = +1; break;
        case Qt::Key_S: movingForward = -1; break;

        case Qt::Key_D: movingRight = +1; break;
        case Qt::Key_A: movingRight = -1; break;

        case Qt::Key_Q: movingUp = +1; break;
        case Qt::Key_E: movingUp = -1; break;

        case Qt::Key_0:
            activeCamera->setPosition(QVector3D(0.0f, 1.0f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_1:
            activeCamera->setPosition(QVector3D(-1.5f, -0.4f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_2:
            activeCamera->setPosition(QVector3D(0.0f, 0.0f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_3:
            activeCamera->setPosition(QVector3D(1.5f, -0.4f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_4:
            activeCamera->setPosition(QVector3D(-2.0f, 0.0f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_5:
            activeCamera->setPosition(QVector3D(0.0f, 2.5f, 0.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            activeCamera->yaw(-90.0f);
            break;
        case Qt::Key_6:
            activeCamera->setPosition(QVector3D(2.0f, 0.0f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_7:
            activeCamera->setPosition(QVector3D(-1.5f, 1.5f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_8:
            activeCamera->setPosition(QVector3D(0.0f, 0.0f, -2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_9:
            activeCamera->setPosition(QVector3D(1.5f, 1.5f, 2.0f));
            activeCamera->setTarget(QVector3D(0.0f, 0.0f, 0.0f));
            break;
        }
    }

    void CameraControl::handleKeyReleased(QKeyEvent* evt) {
        switch (evt->key()) {
        case Qt::Key_W:
        case Qt::Key_S:
            movingForward = 0;
            break;

        case Qt::Key_D:
        case Qt::Key_A:
            movingRight = 0;
            break;

        case Qt::Key_Q:
        case Qt::Key_E:
            movingUp = 0;
            break;
        }
    }
}
