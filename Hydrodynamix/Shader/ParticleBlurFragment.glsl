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

// Parameters from the vertex shader
in vec2 coords;

// Uniforms
uniform sampler2D blurTexture;
uniform vec2 blurDirection;

layout(row_major) uniform GlobalParams{
    mat4 matView;
    mat4 matProj;
    vec4 eyePosition;
    vec4 mousePosition;
    vec4 viewport;
} globalParams;

// Constants
const int gaussRadius = 11;

const float gaussFilter[] = {
    0.0402f, 0.0623f, 0.0877f, 0.1120f, 0.1297f,
    0.1362f, 0.1297f, 0.1120f, 0.0877f, 0.0623f,
    0.0402f
};

// Output
out float fragBlur;

void main() {
    // Half-resolution
    vec2 screenSize = globalParams.viewport.zw / 2.0f;
    vec2 step = blurDirection / screenSize;
    vec2 position = coords - float(int(gaussRadius / 2.0f)) * step;

    // Gaussian Blur
    fragBlur = 0.0f;
    for (int i = 0; i < gaussRadius; ++i) {
        fragBlur += gaussFilter[i] * texture(blurTexture, position).r;
        position += step;
    }
}
