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
#include "UserInteractionManager.hpp"

namespace IO {
    UserInteractionManager::UserInteractionManager(QObject* parent)
        : AbstractInteractionManager(parent) {

    }

    void UserInteractionManager::initialize() {
        for (auto& interaction : interactions) {
            interaction->initialize(this);
        }
    }

    void UserInteractionManager::cleanup() {
        for (auto& interaction : interactions) {
            interaction->cleanup();
        }
    }

    void UserInteractionManager::update() {
        for (auto& interaction : interactions) {
            if (interaction->isRunning()) {
                interaction->process();
            }
        }
    }

    void UserInteractionManager::updateProperties(Data::SimulationConfig* config) {
        for (auto& interaction : interactions) {
            interaction->updateProperties(config);
        }
    }

    void UserInteractionManager::setInputManager(InputManager* inputManager) {
        for (auto& interaction : interactions) {
            interaction->setInputManager(inputManager);
        }
    }
}
