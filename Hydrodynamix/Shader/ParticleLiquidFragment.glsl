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

// Textures
uniform sampler2D particleTexture;
uniform sampler2D thicknessTexture;

layout(row_major) uniform GlobalParams {
    mat4 matView;
    mat4 matProj;
    vec4 eyePosition;
    vec4 mousePosition;
    vec4 viewport;
} globalParams;

// Output
out vec4 fragColor;

// Constants
const float shininess = 30.0f;

const vec3 lightDir = normalize(vec3(1, 1, 1));
const vec3 lightAttenuation = vec3(0.6f, 0.2f, 0.05f);

vec2 spheremap(vec3 dir) {
    float m = 2.0f * sqrt(dir.x * dir.x + dir.y * dir.y + (dir.z + 1.0f) * (dir.z + 1.0f));
    return vec2(dir.x / m + 0.5f, dir.y / m + 0.5f);
}

vec3 getEyespacePos(vec2 coords, float depth) {
    vec2 pos = (coords - 0.5f) * 2.0f;
    float wx = pos.x * -globalParams.matProj[0][0];
    float wy = pos.y * -globalParams.matProj[1][1];
    return depth * vec3(wx, wy, 1.0f);
}

// Compute eye-space normal. Adapted from PySPH.
vec3 getEyespaceNormal(vec2 pos, float depth) {
    vec2 screenSize = globalParams.viewport.zw;

    // Width of one pixel
    vec2 dx = vec2(1.0f / screenSize.x, 0.0f);
    vec2 dy = vec2(0.0f, 1.0f / screenSize.y);

    // Depth derivatives. We could use dFdx, dFdy here but this gives better results.
    float zdxp = texture(particleTexture, pos + dx).r;
    float zdxn = texture(particleTexture, pos - dx).r;
    float zdx = 0.5f * (zdxp - zdxn);

    float zdyp = texture(particleTexture, pos + dy).r;
    float zdyn = texture(particleTexture, pos - dy).r;
    float zdy = 0.5f * (zdyp - zdyn);

    // Projection inversion
    float cx = -2.0f / (screenSize.x * globalParams.matProj[0][0]);
    float cy = -2.0f / (screenSize.y * globalParams.matProj[1][1]);

    // Screenspace coordinates
    float sx = floor(pos.x * (screenSize.x - 1.0f));
    float sy = floor(pos.y * (screenSize.y - 1.0f));
    float wx = (screenSize.x - 2.0f * sx) / (screenSize.x * globalParams.matProj[0][0]);
    float wy = (screenSize.y - 2.0f * sy) / (screenSize.y * globalParams.matProj[1][1]);

    // Eyespace position derivatives
    vec3 pdx = vec3(cx * depth + wx * zdx, wy * zdx, zdx);
    vec3 pdy = vec3(wx * zdy, cy * depth + wy * zdy, zdy);

    return normalize(cross(pdx, pdy));
}

// Compute view-space lighting
vec3 getLightColor(vec3 normal, vec3 eyePos, float thickness) {
    vec3 ld = mat3(globalParams.matView) * lightDir;

    // Lamberts cosine law
    float ndl = dot(normal, ld);
    vec3 r = normalize(ld - 2.0f * ndl * normal);
    vec3 v = normalize(-eyePos);

    float diffuse = abs(ndl) * 0.5f + 0.5f;
    float specular = max(pow(dot(r, v), shininess), 0);

    // Schlicks approximation (from Wikipedia)
    float fresnel = pow(1.0f - abs(dot(normal, -v)), 5.0f);
    float reflection = specular + (1.0f - specular) * fresnel;
    reflection = reflection * clamp(thickness * 0.5f, 0, 1);

    // Beers law
    vec3 beer = exp(-lightAttenuation * thickness * 5.0f);
    return beer * diffuse + vec3(reflection);
}

void main() {
    vec2 particleDepth = texture(particleTexture, coords).rg;
    if (particleDepth.x < 0.0001f) discard;

    vec3 eyePos = getEyespacePos(coords, particleDepth.x);
    vec3 normal = getEyespaceNormal(coords, particleDepth.x);

    float thickness = texture(thicknessTexture, coords).r;
    vec3 light = getLightColor(normal, eyePos, thickness);
    fragColor = vec4(light, 1.0f - exp(-2.0f * thickness));

    // Transform device depth into depth range
    float far = gl_DepthRange.far;
    float near = gl_DepthRange.near;
    float deviceDepth = particleDepth.x / particleDepth.y;
    gl_FragDepth = (deviceDepth * (far - near) + near + far) * 0.5f;
}
