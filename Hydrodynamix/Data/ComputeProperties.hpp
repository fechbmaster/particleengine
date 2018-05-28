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

#ifndef COMPUTE_PROPERTIES
#define COMPUTE_PROPERTIES

#include <cstdint>

namespace Data {
    struct ComputeProperties {
        float worldSizeHalfX;
        float worldSizeHalfY;
        float worldSizeHalfZ;

        float cellSize;
        uint32_t numCells;

        float mass;

        float surfaceTensionThreshold;
        float surfaceTensionThreshold2;

        // Kernel radius
        float smoothingLength;
        float smoothingLength2;
        float smoothingLength3;

        float defaultKernelCoefficient;
        float defaultKernelGradientCoefficient;
        float defaultKernelLaplacianCoefficient;

        float pressureKernelCoefficient;
        float pressureKernelGradientCoefficient;
        float pressureKernelLaplacianCoefficient;

        float viscosityKernelCoefficient;
        float viscosityKernelGradientCoefficient;
        float viscosityKernelLaplacianCoefficient;
    };
}

#endif
