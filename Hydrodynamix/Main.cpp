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
#include "MainWindow.hpp"

#include "Computation/CudaHelper.hpp"

#include <QApplication>
#include <QSurfaceFormat>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

bool applyStyleSheet(const QString& fileName) {
    QFile stylesheet(fileName);
    if (!stylesheet.open(QFile::ReadOnly)) {
        qDebug() << "Unable to load stylesheet file.";
        return false;
    }

    qApp->setStyleSheet(stylesheet.readAll());
    return true;
}

int main(int argc, char *argv[]) {
#if defined(_MSC_VER) && defined(_DEBUG)
    // Enable memory leak detection in debug mode (Visual Studio only)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    // Check for CUDA memory leaks
    atexit([] { Computation::checkForDeviceMemoryLeaks(); });

    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setSwapInterval(1);
    format.setSamples(4);
    format.setVersion(4, 3);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);

    QSurfaceFormat::setDefaultFormat(format);
    QApplication::setDesktopSettingsAware(false);

    QApplication app(argc, argv);
    applyStyleSheet("hydrodynamix.qss");

    MainWindow window;
    window.show();
    return app.exec();
}
