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

#ifndef INPUT_MANAGER_HPP
#define INPUT_MANAGER_HPP

#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QDragEnterEvent>
#include <QDropEvent>

namespace IO {
    class InputManager : public QObject {
        Q_OBJECT

    public:
        InputManager(QObject* parent = nullptr)
            : QObject(parent) {}

        ~InputManager() = default;

    public:
        void handleKeyPressed(QKeyEvent* event);
        void handleKeyReleased(QKeyEvent* event);

        void handleMousePressed(QMouseEvent* event);
        void handleMouseReleased(QMouseEvent* event);
        void handleMouseMoved(QMouseEvent* event);

        void handleMouseWheel(QWheelEvent* event);

        void handleDragEnter(QDragEnterEvent* event);
        void handleDrop(QDropEvent* event);

    signals:
        void keyPressedEvent(QKeyEvent*);
        void keyReleasedEvent(QKeyEvent*);

        void mousePressedEvent(QMouseEvent*);
        void mouseReleasedEvent(QMouseEvent*);
        void mouseMovedEvent(QMouseEvent*);

        void mouseWheelEvent(QWheelEvent*);

        void dragEnterEvent(QDragEnterEvent*);
        void dropEvent(QDropEvent*);
    };
}

#endif
