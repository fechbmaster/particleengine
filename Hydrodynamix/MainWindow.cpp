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
#include "Core/ParticleEngine.hpp"
#include "Scene/ParticleRender.hpp"

#define FOR_EACH_SLIDER_PROPERTY(macro) \
    macro(volume);                      \
    macro(timeStep);                    \
    macro(elasticity);                  \
    macro(gravity);                     \
    macro(mouseForce);                  \
    macro(gasStiffness);                \
    macro(viscosity);                   \
    macro(surfaceTension)

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , engine(new Core::ParticleEngine) {

    ui.setupUi(this);
    updateProperties();

    ui.openGLWidget->setEngine(engine);
    ui.settingsFrame->setHidden(true);
    ui.infoFrame->setHidden(true);

    auto config = engine->getSimulationConfig();
    connect(config, &Data::SimulationConfig::generalPropertiesChanged, this, &MainWindow::updateProperties);
    connect(config, &Data::SimulationConfig::computePropertiesChanged, this, &MainWindow::updateProperties);
    connect(config, &Data::SimulationConfig::physicalPropertiesChanged, this, &MainWindow::updateProperties);
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    int width = event->size().width();
    int height = event->size().height();

    ui.openGLWidget->resize(frameSize());

    int settingsFrameMargin = 50;
    int settingsFrameHeight = std::min(height - settingsFrameMargin, 710);
    ui.settingsFrame->setFixedHeight(settingsFrameHeight);

    int infoFrameMargin = 10;
    int infoFramePosition = width - ui.infoFrame->size().width();
    ui.infoFrame->move(infoFramePosition - infoFrameMargin, infoFrameMargin);
}

void MainWindow::updateSliders() {
    auto config = engine->getSimulationConfig();
    auto physicalProps = config->getPhysicalProperties();

    #define UPDATE_PHYSICAL_SLIDER_VALUE(prop)           \
        ui.prop##Slider->setSliderPosition(uint(physicalProps.prop * 1000))

    FOR_EACH_SLIDER_PROPERTY(UPDATE_PHYSICAL_SLIDER_VALUE);

    uint iterations = engine->getParticleRender()->getSmoothingIterations();
    ui.smoothingIterationsSlider->setSliderPosition(iterations);
}

void MainWindow::updateInfoValueLabels() {
    auto simulationConfig = engine->getSimulationConfig();
    auto computeProps = simulationConfig->getComputeProperties();
    auto generalProps = simulationConfig->getGeneralProperties();

    #define UPDATE_COMPUTE_LABEL_VALUE(prop) \
        ui.prop##Value->setText(QString::number(computeProps.prop))

    #define UPDATE_GENERAL_LABEL_VALUE(prop) \
        ui.prop##Value->setText(QString::number(generalProps.prop))

    UPDATE_GENERAL_LABEL_VALUE(numParticles);
    UPDATE_GENERAL_LABEL_VALUE(particleRadius);
    UPDATE_GENERAL_LABEL_VALUE(worldSizeX);
    UPDATE_GENERAL_LABEL_VALUE(worldSizeY);
    UPDATE_GENERAL_LABEL_VALUE(worldSizeZ);

    UPDATE_COMPUTE_LABEL_VALUE(mass);
    UPDATE_COMPUTE_LABEL_VALUE(smoothingLength);
}

void MainWindow::updateSettingsValueLabels() {
    auto simulationConfig = engine->getSimulationConfig();
    auto physicalProps = simulationConfig->getPhysicalProperties();

    #define UPDATE_PHYSICAL_LABEL_VALUE(prop, unit) \
        ui.prop##Value->setText(QString::number(physicalProps.prop) + " " + unit)

    UPDATE_PHYSICAL_LABEL_VALUE(volume, "m<sup>3</sup>");
    UPDATE_PHYSICAL_LABEL_VALUE(timeStep, "s");
    UPDATE_PHYSICAL_LABEL_VALUE(elasticity, "");
    UPDATE_PHYSICAL_LABEL_VALUE(gravity, "m/s<sup>2</sup>");
    UPDATE_PHYSICAL_LABEL_VALUE(mouseForce, "x");
    UPDATE_PHYSICAL_LABEL_VALUE(gasStiffness, "x");
    UPDATE_PHYSICAL_LABEL_VALUE(viscosity, "Pa*s");
    UPDATE_PHYSICAL_LABEL_VALUE(surfaceTension, "N/m");

    uint iterations = engine->getParticleRender()->getSmoothingIterations();
    ui.smoothingIterationsValue->setText(QString::number(iterations) + " x");
}

void MainWindow::updateProperties() {
    updateSliders();
    updateInfoValueLabels();
    updateSettingsValueLabels();
}

void MainWindow::toggleSettingsFrame() {
    bool hidden = ui.settingsFrame->isHidden();
    ui.settingsFrame->setHidden(!hidden);

    if (hidden) {
        ui.settingsButton->setText("Close Settings");
    } else {
        ui.settingsButton->setText("Open Settings");
    }
}

void MainWindow::toggleInfoFrame() {
    bool hidden = ui.infoFrame->isHidden();
    ui.infoFrame->setHidden(!hidden);

    if (hidden) {
        ui.infoButton->setText("Close Info");
    } else {
        ui.infoButton->setText("Open Info");
    }
}

void MainWindow::resetParams() {
    engine->reloadConfigs();
    updateProperties();
}

void MainWindow::toggleRenderingMode(bool checked) {
    engine->getParticleRender()->setRenderingMode(checked
        ? Scene::ParticleRender::RenderSurface
        : Scene::ParticleRender::RenderSpheres);
}

#define IMPLEMENT_PROPERTY_SLIDER_SLOT(prop)                    \
    void MainWindow::prop##SliderChanged(int value) {           \
        auto config = engine->getSimulationConfig();            \
        auto physicalProps = config->getPhysicalProperties();   \
        physicalProps.prop = value / 1000.0f;                   \
        config->setPhysicalProperties(physicalProps);           \
    }

FOR_EACH_SLIDER_PROPERTY(IMPLEMENT_PROPERTY_SLIDER_SLOT);

void MainWindow::smoothingIterationsSliderChanged(int value) {
    engine->getParticleRender()->setSmoothingIterations(value);
    updateProperties();
}
