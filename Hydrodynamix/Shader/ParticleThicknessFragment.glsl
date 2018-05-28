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

in VertexOutput {
    vec3 worldPos;
    vec3 viewPos;
    flat float velocity;
} fragInput;

out float fragThickness;

uniform float radius;

void main() {
    // gl_PointCoord is in range [0,1] - (0,0) is bottom left
    vec2 pos = gl_PointCoord * 2.0f - 1.0f;

    // Calculate distance from circle center
    float dist2 = dot(pos.xy, pos.xy);
    if (dist2 > 1.0f) discard;

    // Additive-Blend to get a sort-of thickness
    fragThickness = 0.05f * exp(-dist2 * 2.0f);
}
