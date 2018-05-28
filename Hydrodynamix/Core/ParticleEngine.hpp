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

#ifndef PARTICLE_ENGINE_HPP
#define PARTICLE_ENGINE_HPP

#include "AbstractEngine.hpp"

#include "IO/ConfigFile.hpp"
#include "IO/CameraControl.hpp"
#include "IO/InputManager.hpp"

#include "Scene/ModelManager.hpp"
#include "Scene/AbstractScene.hpp"
#include "Scene/ParticleRender.hpp"
#include "Scene/SimulationRoomRender.hpp"

#include "Computation/SolverManager.hpp"
#include "Computation/SPHSolver.hpp"
#include "Computation/GridSolver.hpp"

#include "IO/AbstractInteractionManager.hpp"
#include "IO/WaterBlockInteraction.hpp"
#include "IO/WaterJetInteraction.hpp"

#include "Data/SimulationConfig.hpp"

namespace Core {
    class ParticleEngine : public AbstractEngine {
        Q_OBJECT

    public:
        ParticleEngine(QObject* parent = nullptr);
        ~ParticleEngine() override = default;

    public:
        void initialize() override;
        void cleanup() override;
        void update() override;

        void resize(int width, int height) override;

        void setInputManager(IO::InputManager* inputManager) override;

    public:
        void reloadConfigs();

        Data::SimulationConfig* getSimulationConfig() {
            return &simulationConfig;
        }

    private slots:
        void handleMouseMoved(QMouseEvent* event);

    private:
        void updateProperties(Data::SimulationConfig* config);

    private:
        Scene::ParticleRenderPtr particleRender;
        Scene::SimulationRoomRenderPtr simulationRoomRender;

        Computation::SPHSolverPtr sphSolver;
        Computation::GridSolverPtr gridSolver;

        IO::WaterBlockInteractionPtr waterBlockInteraction;
        IO::WaterJetInteractionPtr waterJetInteraction;

        IO::ConfigFile configFile;
        Data::SimulationConfig simulationConfig;

        QScopedPointer<IO::CameraControl> cameraControl;
        QScopedPointer<Scene::AbstractScene> activeScene;

        QScopedPointer<Scene::ModelManager> modelManager;
        QScopedPointer<Computation::SolverManager> solverManager;
        QScopedPointer<IO::AbstractInteractionManager> interactionManager;

    public:
        DEF_GETTER(getParticleRender, particleRender);
    };
}

#endif
