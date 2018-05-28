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

#ifndef SPH_SOLVER_HPP
#define SPH_SOLVER_HPP

#include "AbstractSolver.hpp"

namespace Computation {
    class SPHSolver : public AbstractSolver {
        Q_OBJECT

    public:
        SPHSolver(QObject* parent = nullptr)
            : AbstractSolver(parent) {}

        ~SPHSolver() override = default;

    public:
        void initialize(SolverManager* manager) override;
        void cleanup() override;
        void compute() override;

        void updateProperties(Data::SimulationConfig* config) override;

    public:
        void handleExternalForces(const QVector3D& source);

    private:
        SolverManager* solverManager = nullptr;
    };

    typedef QSharedPointer<SPHSolver> SPHSolverPtr;
}

Q_DECLARE_METATYPE(Computation::SPHSolverPtr);

#endif
