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

#ifndef MESH_LOADER_THREAD_HPP
#define MESH_LOADER_THREAD_HPP

#include "ModelFile.hpp"

#include <QThread>
#include <QMutex>
#include <QStringList>

namespace IO {
    class ModelLoaderThread : public QThread {
        Q_OBJECT

    public:
        ModelLoaderThread(QObject* parent = nullptr)
            : QThread(parent) {}

        virtual ~ModelLoaderThread() = default;

    public:
        void initialize();
        void shutdown();

        void pushLoadRequest(const QString& fileName);

    private:
        void run() override;
        void processLoadRequests();

    signals:
        void modelLoaded(const IO::ModelFilePtr&);
        void modelLoadFailed(const QString& fileName);

    private:
        QStringList loadRequests;
        QMutex loaderLock;

        volatile bool shouldRun = false;
    };
}

#endif
