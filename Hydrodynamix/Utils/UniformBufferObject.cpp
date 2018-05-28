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
#include "UniformBufferObject.hpp"

#include <QGLContext>
#include <QDebug>

namespace Utils {
    QVector<GLuint> UniformBufferObject::bindingIndices;

    GLuint UniformBufferObject::bindingIndex() {
        for (GLuint i = 0, size = bindingIndices.size(); i < size; i++) {
            if (bindingIndices.at(i) != i) {
                bindingIndices.insert(i, i);
                return i;
            }
        }

        bindingIndices.append(bindingIndices.size());
        return bindingIndices.size();
    }

    bool UniformBufferObject::create() {
        glContext = QOpenGLContext::currentContext();

        if (glContext) {
            if (!isInitialized && !initFunctionPointers(glContext)) {
                qDebug("Cannot find Uniform Buffer Objects related functions");
                return false;
            }

            GLuint tmpBufferId = 0;
            glGenBuffers(1, &tmpBufferId);

            if (tmpBufferId) {
                uniformBufferId = tmpBufferId;
                return true;
            }
            else {
                qDebug("Invalid buffer Id");
            }
        }

        qDebug("Could not retrieve buffer");
        return false;
    }

    bool UniformBufferObject::isCreated() const {
        return uniformBufferId != 0;
    }

    void UniformBufferObject::destroy() {
        if (uniformBufferId != 0) {
            glDeleteBuffers(1, &uniformBufferId);
        }
        uniformBufferId = 0;
        glContext = nullptr;
    }

    bool UniformBufferObject::bind() {
        if (!isCreated()) {
            qDebug("Buffer not created");
            return false;
        }

        glBindBuffer(GL_UNIFORM_BUFFER, uniformBufferId);
        return true;
    }

    bool UniformBufferObject::bind(GLuint progId, const QString& uniformName) {
        GLuint tmpBlockIdx = glGetUniformBlockIndex(progId, uniformName.toUtf8());

        if (tmpBlockIdx == GL_INVALID_INDEX) {
            qDebug() << QString("Could not find block index of block named: %1").arg(uniformName);
            return false;
        }

        GLint tmpBlockSize;
        glGetActiveUniformBlockiv(progId, tmpBlockIdx, GL_UNIFORM_BLOCK_DATA_SIZE, &tmpBlockSize);

        GLuint tmpBindingIdx = bindingIndex();
        glUniformBlockBinding(progId, tmpBlockIdx, tmpBindingIdx);

        glBindBufferBase(GL_UNIFORM_BUFFER, tmpBindingIdx, uniformBufferId);
        if (glGetError() == GL_INVALID_VALUE || glGetError() == GL_INVALID_ENUM) {
            qDebug() << "ERROR";
        }

        BufferInfo info;
        info.progId = progId;
        info.uniformName = uniformName;
        info.blockIndex = tmpBlockIdx;
        info.blockSize = tmpBlockSize;
        info.bindingIndex = tmpBindingIdx;

        uniformBufferInfos.append(info);
        return true;
    }

    void UniformBufferObject::release() {
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

    GLuint UniformBufferObject::bufferId() const {
        return uniformBufferId;
    }

    void UniformBufferObject::write(int offset, const void *data, int count) {
        if (!isCreated()) {
            return;
        }

        bind();
        glBufferSubData(GL_UNIFORM_BUFFER, offset, count, data);
    }

    void UniformBufferObject::allocate(const void *data, int count) {
        if (!isCreated()) {
            return;
        }

        bind();
        glBufferData(GL_UNIFORM_BUFFER, count, data, GL_DYNAMIC_DRAW);
    }

    bool UniformBufferObject::initFunctionPointers(const QOpenGLContext* glContext) {
        glBindBuffer = (PFNGLBINDBUFFERPROC)glContext->getProcAddress("glBindBuffer");
        glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)glContext->getProcAddress("glBindBufferBase");
        glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)glContext->getProcAddress("glBindBufferRange");
        glBufferData = (PFNGLBUFFERDATAPROC)glContext->getProcAddress("glBufferData");
        glBufferSubData = (PFNGLBUFFERSUBDATAPROC)glContext->getProcAddress("glBufferSubData");
        glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glContext->getProcAddress("glDeleteBuffers");
        glGenBuffers = (PFNGLGENBUFFERSPROC)glContext->getProcAddress("glGenBuffers");
        glGetActiveUniformBlockiv = (PFNGLGETACTIVEUNIFORMBLOCKIVPROC)glContext->getProcAddress("glGetActiveUniformBlockiv");
        glGetActiveUniformsiv = (PFNGLGETACTIVEUNIFORMSIVPROC)glContext->getProcAddress("glGetActiveUniformsiv");
        glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)glContext->getProcAddress("glGetUniformBlockIndex");
        glGetUniformIndices = (PFNGLGETUNIFORMINDICESPROC)glContext->getProcAddress("glGetUniformIndices");
        glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC)glContext->getProcAddress("glUniformBlockBinding");

        if (!glBindBuffer
            || !glBindBufferBase
            || !glBindBufferRange
            || !glBufferData
            || !glBufferSubData
            || !glDeleteBuffers
            || !glGenBuffers
            || !glGetActiveUniformBlockiv
            || !glGetActiveUniformsiv
            || !glGetUniformBlockIndex
            || !glGetUniformIndices
            || !glUniformBlockBinding) {

            qDebug("Could not init function pointers");
            return false;
        }

        return true;
    }
}
