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

#ifndef SIMPLE_BATCH_RENDER_HPP
#define SIMPLE_BATCH_RENDER_HPP

#include "AbstractRender.hpp"

namespace Scene {
    // This class combines one or more AbstractRenderers into a single one (grouping).
    class SimpleBatchRender : public AbstractRender {
        Q_OBJECT

    public:
        SimpleBatchRender(QObject* parent = nullptr);
        ~SimpleBatchRender() override = default;

    public:
        void initialize(AbstractScene* parent) override;
        void cleanup() override;
        void update() override;
        void render() override;

    public:
        void addRenderer(const AbstractRenderPtr& renderer);

    private:
        AbstractScene* parentScene = nullptr;

        QVector<AbstractRenderPtr> renderers;
        QVector<AbstractRenderPtr> newRenderers;
    };

    typedef QSharedPointer<SimpleBatchRender> SimpleBatchRenderPtr;
}

Q_DECLARE_METATYPE(Scene::SimpleBatchRenderPtr);

#endif
