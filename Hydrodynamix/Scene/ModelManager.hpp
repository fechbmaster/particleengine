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

#ifndef MODEL_MANAGER_HPP
#define MODEL_MANAGER_HPP

#include "AbstractScene.hpp"
#include "SimpleBatchRender.hpp"
#include "ModelInstance.hpp"
#include "ModelRender.hpp"

#include "Data/SimulationConfig.hpp"

#include "IO/ModelLoaderThread.hpp"
#include "IO/InputManager.hpp"

#include <QHash>

namespace Scene {
    class ModelManager : public QObject {
        Q_OBJECT

    public:
        ModelManager(QObject* parent = nullptr);
        ~ModelManager() = default;

        void initialize(AbstractScene* scene);
        void cleanup();
        void update();

        void addModel(const QString& fileName);

        void updateProperties(Data::SimulationConfig* config);
        void setInputManager(IO::InputManager* inputManager);

    public:
        AbstractRenderPtr getModelRender() const {
            return modelBatchRender;
        }

    private slots:
        void modelLoaded(const IO::ModelFilePtr& file);
        void modelLoadFailed(const QString& fileName);

        void handleMousePressed(QMouseEvent* evt);
        void handleMouseWheel(QWheelEvent* evt);
        void handleKeyPressed(QKeyEvent* evt);
        void handleKeyReleased(QKeyEvent* evt);

        void handleDragEnter(QDragEnterEvent* event);
        void handleDrop(QDropEvent* event);

    private:
        AbstractScene* worldScene = nullptr;

        SimpleBatchRenderPtr modelBatchRender;
        IO::ModelLoaderThread modelLoaderThread;

        ModelInstancePtr activeInstance;
        QHash<QString, ModelRenderPtr> modelRenderers;

        int activeKey = 0;
    };
}

#endif
