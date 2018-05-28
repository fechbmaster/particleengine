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

in vec4 position0;

out VertexOutput {
    vec3 worldPos;
    vec3 viewPos;
    flat float velocity;
} vertexOutput;

uniform float radius;

// Particles are rendered as 2D points
// The vertex shader needs to scale the points based on distance from camera
void main() {
    vertexOutput.worldPos = position0.xyz;
    vertexOutput.velocity = position0.w;

    vec4 viewPos = vec4(position0.xyz, 1.0f);
    viewPos = globalParams.matView * viewPos;
    vertexOutput.viewPos = viewPos.xyz;

    vec4 projPos = vec4(radius, radius, viewPos.zw);
    projPos = globalParams.matProj * projPos;

    vec2 screenSize = globalParams.viewport.zw;
    vec2 projSize = (screenSize * projPos.xy) / projPos.w;

    gl_PointSize = 0.25f * (projSize.x + projSize.y);
    gl_Position = globalParams.matProj * viewPos;
}
