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

#ifndef OPEN_GL_FUNCTIONS_HPP
#define OPEN_GL_FUNCTIONS_HPP

#include <QOpenGLFunctions_4_3_Core>

namespace Utils {
    class OpenGLFunctions : public QOpenGLFunctions_4_3_Core {
    public:
        OpenGLFunctions() = default;
        ~OpenGLFunctions() = default;

    public:
        GLuint makeBuffer(GLenum type, const GLvoid* data, GLsizeiptr size, GLenum usage);
        GLuint makeTextureBuffer(GLsizei w, GLsizei h, GLenum format, GLint internalFormat);
        GLuint makeFramebuffer(GLsizei w, GLsizei h, GLenum format, GLint internalFormat, GLuint* texture, GLuint depthBuffer);
    };
}

#endif
