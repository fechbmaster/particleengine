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

#ifndef PERSPECTIVE_CAMERA_HPP
#define PERSPECTIVE_CAMERA_HPP

#include "Camera.hpp"

namespace Scene {
    class PerspectiveCamera : public Camera {
    public:
        PerspectiveCamera();
        ~PerspectiveCamera() override = default;

    public:
        virtual void setClip(float znear, float zfar);
        virtual void setAspect(float aspect);
        virtual void setFieldOfView(float fov);

    private:
        void updateProjection();

    private:
        QMatrix4x4 matProjection;
        QMatrix4x4 matProjInverse;

        float aspectRatio = 1.0f;
        float fieldOfView = 45.0f;

        float nearClip = 0.05f;
        float farClip = 100.0f;

    public:
        DEF_GETTER(getProjMatrix, matProjection);
        DEF_GETTER(getProjInverse, matProjInverse);

        DEF_GETTER(getAspect, aspectRatio);
        DEF_GETTER(getFieldOfView, fieldOfView);
        DEF_GETTER(getNearClip, nearClip);
        DEF_GETTER(getFarClip, farClip);
    };
}

#endif
