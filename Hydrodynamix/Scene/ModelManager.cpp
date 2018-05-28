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
#include "ModelManager.hpp"

namespace Scene {
    ModelManager::ModelManager(QObject* parent)
        : QObject(parent)
        , modelBatchRender(new Scene::SimpleBatchRender) {

    }

    void ModelManager::initialize(AbstractScene* scene) {
        worldScene = scene;
        worldScene->addRenderer(modelBatchRender);

        connect(&modelLoaderThread, &IO::ModelLoaderThread::modelLoaded,
            this, &ModelManager::modelLoaded, Qt::QueuedConnection);

        connect(&modelLoaderThread, &IO::ModelLoaderThread::modelLoadFailed,
            this, &ModelManager::modelLoadFailed, Qt::QueuedConnection);

        modelLoaderThread.initialize();
    }

    void ModelManager::cleanup() {
        modelLoaderThread.shutdown();
        modelBatchRender.clear();
    }

    void ModelManager::update() {
        if (activeInstance) {
            activeInstance->moveTo(worldScene->getMousePosition());
        }
    }

    void ModelManager::updateProperties(Data::SimulationConfig* config) {
        // TODO: Pass to ModelLoaderThread/ModelFile
    }

    void ModelManager::setInputManager(IO::InputManager* inputManager) {
        connect(inputManager, &IO::InputManager::mouseWheelEvent, this, &ModelManager::handleMouseWheel);
        connect(inputManager, &IO::InputManager::mousePressedEvent, this, &ModelManager::handleMousePressed);
        connect(inputManager, &IO::InputManager::keyPressedEvent, this, &ModelManager::handleKeyPressed);
        connect(inputManager, &IO::InputManager::keyReleasedEvent, this, &ModelManager::handleKeyReleased);
        connect(inputManager, &IO::InputManager::dragEnterEvent, this, &ModelManager::handleDragEnter);
        connect(inputManager, &IO::InputManager::dropEvent, this, &ModelManager::handleDrop);
    }

    void ModelManager::addModel(const QString& fileName) {
        // Check if it's a model that hasn't been loaded yet
        if (!modelRenderers.contains(fileName)) {
            modelLoaderThread.pushLoadRequest(fileName);
        }

        // Check if we have an existing renderer for this model
        if (ModelRenderPtr renderer = modelRenderers[fileName]) {
            activeInstance.reset(new ModelInstance(renderer->getModelFile()));
            renderer->addRenderInstance(activeInstance);
        }
    }

    void ModelManager::modelLoaded(const IO::ModelFilePtr& file) {
        ModelRenderPtr renderer(new ModelRender(file));
        modelRenderers[file->getFileName()] = renderer;
        modelBatchRender->addRenderer(renderer);

        activeInstance.reset(new ModelInstance(file));
        renderer->addRenderInstance(activeInstance);
    }

    void ModelManager::modelLoadFailed(const QString& fileName) {
        qDebug() << "Could not load model:" << fileName;

        // Remove model from renderers dictionary so we can attempt to load it again
        modelRenderers.remove(fileName);
    }

    void ModelManager::handleMousePressed(QMouseEvent* evt) {
        if (evt->buttons() & Qt::LeftButton) {
            activeInstance.clear();
        }
    }

    void ModelManager::handleKeyPressed(QKeyEvent* evt) {
        if (!activeKey) {
            activeKey = evt->key();
        }
    }

    void ModelManager::handleKeyReleased(QKeyEvent* evt) {
        if (activeKey == evt->key()) {
            activeKey = 0;
        }
    }

    void ModelManager::handleMouseWheel(QWheelEvent* evt) {
        if (!activeInstance) {
            return;
        }

        float rotationDelta = evt->delta() / 10.0f;
        float scalingDelta = evt->delta() / 1000.0f;

        switch (activeKey) {
        case Qt::Key_Alt:
            activeInstance->rotate(rotationDelta, QVector3D(0.0f, 0.0f, 1.0f));
            break;
        case Qt::Key_Control:
            activeInstance->rotate(rotationDelta, QVector3D(1.0f, 0.0f, 0.0f));
            break;
        case Qt::Key_Shift:
            activeInstance->scale(scalingDelta);
            break;
        default:
            activeInstance->rotate(rotationDelta, QVector3D(0.0f, 1.0f, 0.0f));
            break;
        }
    }

    void ModelManager::handleDragEnter(QDragEnterEvent* event) {
        if (event->mimeData()->hasUrls()) {
            auto urlList = event->mimeData()->urls();
            event->setAccepted(urlList.size() == 1);
        }
    }

    void ModelManager::handleDrop(QDropEvent* event) {
        QList<QUrl> urlList = event->mimeData()->urls();
        addModel(urlList[0].toLocalFile());
    }
}
