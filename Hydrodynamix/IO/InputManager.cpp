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
#include "InputManager.hpp"

namespace IO {
    void InputManager::handleKeyPressed(QKeyEvent* event) {
        emit keyPressedEvent(event);
    }

    void InputManager::handleKeyReleased(QKeyEvent* event) {
        emit keyReleasedEvent(event);
    }

    void InputManager::handleMousePressed(QMouseEvent* event) {
        emit mousePressedEvent(event);
    }

    void InputManager::handleMouseReleased(QMouseEvent* event) {
        emit mouseReleasedEvent(event);
    }

    void InputManager::handleMouseMoved(QMouseEvent* event) {
        emit mouseMovedEvent(event);
    }

    void InputManager::handleMouseWheel(QWheelEvent* event) {
        emit mouseWheelEvent(event);
    }

    void InputManager::handleDragEnter(QDragEnterEvent* event) {
        emit dragEnterEvent(event);
    }

    void InputManager::handleDrop(QDropEvent* event) {
        emit dropEvent(event);
    }
}
