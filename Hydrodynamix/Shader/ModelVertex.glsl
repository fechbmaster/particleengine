#version 430 core

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

layout(row_major) uniform GlobalParams {
    mat4 matView;
    mat4 matProj;
    vec4 eyePosition;
    vec4 mousePosition;
    vec4 viewport;
} globalParams;

in vec3 position0;
in vec3 normal0;
in vec4 color0;
in mat4 transform0;

uniform mat4 nodeTransform;

out VertexOutput {
    vec3 normal;
    vec4 color;
    vec3 worldPos;
} vertexOutput;

void main() {
    vec3 normal = normal0;
    normal = mat3(nodeTransform) * normal;
    normal = mat3(transform0) * normal;

    vertexOutput.normal = normal;
    vertexOutput.color = color0;

    vec4 position = vec4(position0, 1.0f);
    position = nodeTransform * position;
    position = transform0 * position;
    vertexOutput.worldPos = position.xyz;

    position = globalParams.matView * position;
    position = globalParams.matProj * position;
    gl_Position = position;
}
