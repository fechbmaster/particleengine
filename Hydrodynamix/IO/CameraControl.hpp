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

#ifndef CAMERA_CONTROL_HPP
#define CAMERA_CONTROL_HPP

#include "Scene/Camera.hpp"

#include <QPoint>

namespace IO {
    class InputManager;

    class CameraControl : public QObject {
        Q_OBJECT

    public:
        CameraControl(QObject* parent = nullptr)
            : QObject(parent) {}

        ~CameraControl() = default;

    public:
        void update();

        void setTarget(Scene::Camera* target);
        void setInputManager(InputManager* inputManager);

    private slots:
        void handleMouseMoved(QMouseEvent* event);
        void handleMousePressed(QMouseEvent* event);
        void handleMouseReleased(QMouseEvent* event);

        void handleKeyPressed(QKeyEvent* event);
        void handleKeyReleased(QKeyEvent* event);

    private:
        Scene::Camera* activeCamera = nullptr;

        QPoint lastCursorPos;
        qint64 lastUpdate = 0;

        int movingForward = 0;
        int movingRight = 0;
        int movingUp = 0;
    };
}

#endif
