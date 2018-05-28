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

#ifndef ABSTRACT_INTERACTION_HPP
#define ABSTRACT_INTERACTION_HPP

#include "Data/SimulationConfig.hpp"

namespace IO {
    class InputManager;
    class AbstractInteractionManager;

    class AbstractInteraction : public QObject {
        Q_OBJECT

    public:
        AbstractInteraction(QObject* parent = nullptr)
            : QObject(parent) {}

        virtual ~AbstractInteraction() = default;

    public:
        virtual void initialize(AbstractInteractionManager*) = 0;
        virtual void cleanup() = 0;
        virtual void process() = 0;

        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool isRunning() const = 0;

        virtual void updateProperties(Data::SimulationConfig*) = 0;
        virtual void setInputManager(InputManager* inputManager) = 0;
    };

    typedef QSharedPointer<AbstractInteraction> AbstractInteractionPtr;
}

Q_DECLARE_METATYPE(IO::AbstractInteractionPtr);

#endif
