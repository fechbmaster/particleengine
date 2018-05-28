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

#ifndef KEY_INTERACTION_HPP
#define KEY_INTERACTION_HPP

#include "AbstractInteraction.hpp"

namespace IO {
    class KeyInteraction : public AbstractInteraction {
        Q_OBJECT

    public:
        enum KeyInteractionType {
            PushToToggle,   //< Pressing the trigger key will toggle the interaction
            HoldPressed     //< Holding down the trigger key will run the interaction
        };

    public:
        KeyInteraction(int key, QObject* parent = nullptr);
        ~KeyInteraction() override = default;

        void setInputManager(InputManager* inputManager) override;

    public:
        virtual void setTriggerKey(int key) {
            triggerKey = key;
        }

        virtual int getTriggerKey() const {
            return triggerKey;
        }

        virtual void setInteractionType(KeyInteractionType type) {
            interactionType = type;
        }

        virtual KeyInteractionType getInteractionType() const {
            return interactionType;
        }

    private slots:
        void handleKeyPressed(QKeyEvent* event);
        void handleKeyReleased(QKeyEvent* event);

    private:
        int triggerKey = 0;
        KeyInteractionType interactionType = PushToToggle;
    };

    typedef QSharedPointer<KeyInteraction> KeyInteractionPtr;
}

Q_DECLARE_METATYPE(IO::KeyInteractionPtr);

#endif
