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
#include "ParticleEngine.hpp"

#include "Scene/WorldScene.hpp"

#include "IO/UserInteractionManager.hpp"

#include "Data/GeneralProperties.hpp"
#include "Data/PhysicalProperties.hpp"

namespace Core {
    ParticleEngine::ParticleEngine(QObject* parent)
        : AbstractEngine(parent)
        , activeScene(new Scene::WorldScene)
        , cameraControl(new IO::CameraControl)
        , modelManager(new Scene::ModelManager)
        , particleRender(new Scene::ParticleRender)
        , simulationRoomRender(new Scene::SimulationRoomRender)
        , sphSolver(new Computation::SPHSolver)
        , gridSolver(new Computation::GridSolver)
        , solverManager(new Computation::SolverManager)
        , waterBlockInteraction(new IO::WaterBlockInteraction)
        , waterJetInteraction(new IO::WaterJetInteraction)
        , interactionManager(new IO::UserInteractionManager) {

        modelManager->initialize(activeScene.data());
        activeScene->addRenderer(simulationRoomRender);
        activeScene->addRenderer(particleRender);

        solverManager->addSolver(gridSolver);
        solverManager->addSolver(sphSolver);

        interactionManager->addInteraction(waterBlockInteraction);
        interactionManager->addInteraction(waterJetInteraction);

        cameraControl->setTarget(activeScene->getActiveCamera());

        reloadConfigs();
    }

    void ParticleEngine::initialize() {
        activeScene->initialize();
        solverManager->initialize();
        interactionManager->initialize();

        auto particleBuffer = particleRender->getBuffer();
        solverManager->setParticleBuffer(particleBuffer);
        waterBlockInteraction->setParticleBuffer(particleBuffer);
        waterJetInteraction->setParticleBuffer(particleBuffer);

        auto particleData = solverManager->getParticleData();
        waterJetInteraction->setParticleData(particleData);
    }

    void ParticleEngine::cleanup() {
        solverManager->cleanup();
        activeScene->cleanup();
        modelManager->cleanup();
        interactionManager->cleanup();
    }

    void ParticleEngine::update() {
        solverManager->compute();

        cameraControl->update();
        activeScene->update();
        modelManager->update();
        interactionManager->update();

        activeScene->render();
    }

    void ParticleEngine::resize(int width, int height) {
        activeScene->resize(width, height);
    }

    void ParticleEngine::setInputManager(IO::InputManager* inputManager) {
        modelManager->setInputManager(inputManager);
        cameraControl->setInputManager(inputManager);
        interactionManager->setInputManager(inputManager);

        connect(inputManager, &IO::InputManager::mouseMovedEvent, this, &ParticleEngine::handleMouseMoved);
    }

    void ParticleEngine::handleMouseMoved(QMouseEvent* event) {
        activeScene->updateMousePosition(event->x(), event->y());

        QVector3D mousePosition = activeScene->getMousePosition();
        waterJetInteraction->setJetTarget(mousePosition);

        if (event->buttons() & Qt::LeftButton) {
            sphSolver->handleExternalForces(mousePosition);
        }
    }

    void ParticleEngine::reloadConfigs() {
        #define READ_PROPERTY(prop, value) \
            prop = configFile.read(#prop, value)

        Data::PhysicalProperties physicalProps;
        READ_PROPERTY(physicalProps.volume, 0.5f);
        READ_PROPERTY(physicalProps.timeStep, 0.005f);
        READ_PROPERTY(physicalProps.elasticity, 0.95f);
        READ_PROPERTY(physicalProps.gravity, 9.81f);
        READ_PROPERTY(physicalProps.mouseForce, 5.0f);

        READ_PROPERTY(physicalProps.restDensity, 998.29f);
        READ_PROPERTY(physicalProps.gasStiffness, 3.0f);
        READ_PROPERTY(physicalProps.viscosity, 5.0f);
        READ_PROPERTY(physicalProps.surfaceTension, 0.0728f);

        READ_PROPERTY(physicalProps.kernelParticles, 64);

        Data::GeneralProperties generalProps;
        READ_PROPERTY(generalProps.numParticles, 50000);
        READ_PROPERTY(generalProps.particleRadius, 1.0f / 64.0f);

        READ_PROPERTY(generalProps.worldSizeX, 1.5f);
        READ_PROPERTY(generalProps.worldSizeY, 1.0f);
        READ_PROPERTY(generalProps.worldSizeZ, 1.0f);

        READ_PROPERTY(generalProps.gridSize, 64);

        simulationConfig.setPhysicalProperties(physicalProps);
        simulationConfig.setGeneralProperties(generalProps);

        updateProperties(&simulationConfig);
    }

    void ParticleEngine::updateProperties(Data::SimulationConfig* config) {
        solverManager->updateProperties(config);
        activeScene->updateProperties(config);
        interactionManager->updateProperties(config);
    }
}
