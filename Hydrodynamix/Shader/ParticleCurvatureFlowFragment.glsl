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

in vec2 coords;

// Texture
uniform sampler2D particleTexture;

layout(row_major) uniform GlobalParams {
    mat4 matView;
    mat4 matProj;
    vec4 eyePosition;
    vec4 mousePosition;
    vec4 viewport;
} globalParams;

// Output
out vec2 fragDepth;

// Mean curvature. From "Screen Space Fluid Rendering with Curvature Flow"
vec3 getMeanCurvature(vec2 pos, float zc) {
    // Curvature is drawn in half-resolution
    vec2 screenSize = globalParams.viewport.zw / 2.0f;

    // Width of one pixel
    vec2 dx = vec2(1.0f / screenSize.x, 0.0f);
    vec2 dy = vec2(0.0f, 1.0f / screenSize.y);

    // Depth derivatives. We cannot use dFdx, dFdy here because of second order differences.
    float zdxp = texture(particleTexture, pos + dx).r;
    float zdxn = texture(particleTexture, pos - dx).r;
    float zdx = 0.5f * (zdxp - zdxn);

    float zdyp = texture(particleTexture, pos + dy).r;
    float zdyn = texture(particleTexture, pos - dy).r;
    float zdy = 0.5f * (zdyp - zdyn);

    // Take second order finite differences
    float zdx2 = zdxp + zdxn - 2.0f * zc;
    float zdy2 = zdyp + zdyn - 2.0f * zc;

    // Second order finite differences, alternating variables
    float zdxpyp = texture(particleTexture, pos + dx + dy).r;
    float zdxnyn = texture(particleTexture, pos - dx - dy).r;
    float zdxpyn = texture(particleTexture, pos + dx - dy).r;
    float zdxnyp = texture(particleTexture, pos - dx + dy).r;
    float zdxy = 0.25f * (zdxpyp + zdxnyn - zdxpyn - zdxnyp);

    // Projection transform inversion terms
    float cx = -2.0f / (screenSize.x * globalParams.matProj[0][0]);
    float cy = -2.0f / (screenSize.y * globalParams.matProj[1][1]);

    // Normalization term
    float d = cy * cy * zdx * zdx + cx * cx * zdy * zdy + cx * cx * cy * cy * zc * zc;

    // Derivatives of said term
    float ddx = cy * cy * 2.0f * zdx * zdx2 + cx * cx * 2.0f * zdy * zdxy + cx * cx * cy * cy * 2.0f * zc * zdx;
    float ddy = cy * cy * 2.0f * zdx * zdxy + cx * cx * 2.0f * zdy * zdy2 + cx * cx * cy * cy * 2.0f * zc * zdy;

    // Temporary variables to calculate mean curvature
    float ex = 0.5f * zdx * ddx - zdx2 * d;
    float ey = 0.5f * zdy * ddy - zdy2 * d;

    // Finally, mean curvature
    float h = 0.5f * ((cy * ex + cx * ey) / pow(d, 1.5f));

    return vec3(zdx, zdy, h);
}

void main() {
    const float dt = 0.0003f;
    const float dzt = 1000.0f;

    vec2 particleDepth = texture(particleTexture, coords).rg;

    if (particleDepth.x < 0.0001f) {
        // At boundary - skip this fragment
        fragDepth = particleDepth;
    } else {
        // Vary contribution with absolute depth differential - trick from pySPH
        vec3 dxyz = getMeanCurvature(coords, particleDepth.r);
        fragDepth = particleDepth + dxyz.z * dt * (1.0f + (abs(dxyz.x) + abs(dxyz.y)) * dzt);
    }
}
