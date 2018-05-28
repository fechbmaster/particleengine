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

#ifndef MAIN_WINDOW_HPP
#define MAIN_WINDOW_HPP

#include "ui_hydrodynamix.h"
#include "Core/ParticleEngine.hpp"

#include <QtWidgets/QMainWindow>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

public slots:
    void volumeSliderChanged(int);
    void timeStepSliderChanged(int);
    void elasticitySliderChanged(int);
    void gravitySliderChanged(int);
    void mouseForceSliderChanged(int);
    void gasStiffnessSliderChanged(int);
    void viscositySliderChanged(int);
    void surfaceTensionSliderChanged(int);
    void smoothingIterationsSliderChanged(int);

    void resetParams();
    void toggleSettingsFrame();
    void toggleInfoFrame();
    void toggleRenderingMode(bool checked);
    void updateProperties();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void updateSliders();
    void updateInfoValueLabels();
    void updateSettingsValueLabels();

private:
    QSharedPointer<Core::ParticleEngine> engine;
    Ui::HydrodynamixClass ui;
};

#endif
