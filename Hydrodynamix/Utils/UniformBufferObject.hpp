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

#ifndef UNIFORM_BUFFER_OBJECT_HPP
#define UNIFORM_BUFFER_OBJECT_HPP

#include <qgl.h>
#include <QVector>
#include <QString>

namespace Utils {
    class UniformBufferObject {
    public:
        UniformBufferObject() = default;
        ~UniformBufferObject() = default;

    public:
        bool create();
        bool isCreated() const;

        void destroy();

        bool bind();
        bool bind(GLuint progId, const QString& uniformName);
        void release();

        GLuint bufferId() const;

        void write(int offset, const void* data, int count);
        void allocate(const void* data, int count);

        void allocate(int count) {
            allocate(0, count);
        }

    private:
        bool initFunctionPointers(const QOpenGLContext* context);

    private:
        static GLuint bindingIndex();

    private:
        static QVector<GLuint> bindingIndices;

        struct BufferInfo {
            QString uniformName;
            GLuint progId;
            GLuint blockIndex;
            GLint blockSize;
            GLuint bindingIndex;
        };

        GLuint uniformBufferId = 0;
        const QOpenGLContext* glContext = nullptr;
        bool isInitialized = false;

        QVector<BufferInfo> uniformBufferInfos;

    private:
        PFNGLBINDBUFFERPROC glBindBuffer;
        PFNGLBINDBUFFERBASEPROC glBindBufferBase;
        PFNGLBINDBUFFERRANGEPROC glBindBufferRange;
        PFNGLBUFFERDATAPROC glBufferData;
        PFNGLBUFFERSUBDATAPROC glBufferSubData;
        PFNGLDELETEBUFFERSPROC glDeleteBuffers;
        PFNGLGENBUFFERSPROC glGenBuffers;
        PFNGLGETACTIVEUNIFORMBLOCKIVPROC glGetActiveUniformBlockiv;
        PFNGLGETACTIVEUNIFORMSIVPROC glGetActiveUniformsiv;
        PFNGLGETUNIFORMBLOCKINDEXPROC glGetUniformBlockIndex;
        PFNGLGETUNIFORMINDICESPROC glGetUniformIndices;
        PFNGLUNIFORMBLOCKBINDINGPROC glUniformBlockBinding;
    };
}

#endif
