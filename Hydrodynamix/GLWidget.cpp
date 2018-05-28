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
#include "GLWidget.hpp"

GLWidget::GLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , inputManager(new IO::InputManager) {

    setAcceptDrops(true);
    setMouseTracking(true);
    setFocusPolicy(Qt::WheelFocus);
}

GLWidget::~GLWidget() {
    cleanup();
}

void GLWidget::cleanup() {
    makeCurrent();
    engine->cleanup();
    doneCurrent();
}

void GLWidget::initializeGL() {
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &GLWidget::cleanup);
    engine->initialize();
    engine->setInputManager(inputManager.data());
}

void GLWidget::paintGL() {
    engine->update();
    update();
}

void GLWidget::resizeGL(int w, int h) {
    engine->resize(w, h);
}

void GLWidget::mousePressEvent(QMouseEvent* event) {
    inputManager->handleMousePressed(event);
}

void GLWidget::mouseReleaseEvent(QMouseEvent* event) {
    inputManager->handleMouseReleased(event);
}

void GLWidget::mouseMoveEvent(QMouseEvent* event) {
    inputManager->handleMouseMoved(event);
}

void GLWidget::wheelEvent(QWheelEvent* event) {
    inputManager->handleMouseWheel(event);
}

void GLWidget::keyPressEvent(QKeyEvent* event) {
    inputManager->handleKeyPressed(event);
}

void GLWidget::keyReleaseEvent(QKeyEvent* event) {
    inputManager->handleKeyReleased(event);
}

void GLWidget::dragEnterEvent(QDragEnterEvent* event) {
    inputManager->handleDragEnter(event);
}

void GLWidget::dropEvent(QDropEvent* event) {
    inputManager->handleDrop(event);
}
