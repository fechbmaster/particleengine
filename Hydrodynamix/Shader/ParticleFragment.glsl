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

out vec4 fragColor;

uniform float radius;

const float shininess = 3.0f;
const float lightIntensity = 1.0f;

const vec3 lightDir = normalize(vec3(1, 1, 1));
const vec3 ambientLight = vec3(0.1f, 0.1f, 0.1f);
const vec3 diffuseLight = vec3(1.0f, 1.0f, 1.0f);
const vec3 specularLight = vec3(1.0f, 1.0f, 1.0f);
const vec3 emissiveLight = vec3(0.0f, 0.0f, 0.0f);

vec3 getLightColor(vec3 normal, vec3 worldPos) {
    normal = normalize(normal);

    float ndl = dot(normal, lightDir);
    vec3 r = normalize(lightDir - 2.0f * ndl * normal);
    vec3 v = normalize(worldPos - globalParams.eyePosition.xyz);

    float diffuse = max(ndl * 0.5f + 0.5f, 0);
    float specular = max(pow(dot(r, v), shininess), 0);

    vec3 lightColor = ambientLight;
    lightColor.rgb += diffuseLight * diffuse;
    lightColor.rgb += specularLight * specular;
    return lightColor * lightIntensity + emissiveLight;
}

void main() {
    // gl_PointCoord is in range [0,1] - (0,0) is bottom left
    vec2 pos = gl_PointCoord * 2.0f - 1.0f;

    // Calculate distance from circle center
    float dist2 = dot(pos.xy, pos.xy);
    if (dist2 > 1.0f) discard;

    // Calculate normal vector
    float z = sqrt(1.0f - dist2);
    vec3 normal = vec3(pos.xy, z);

    // Apply depth correction to make our point look like a sphere in the depth buffer
    vec3 viewPos = fragInput.viewPos + normal * radius;
    vec4 clipPos = globalParams.matProj * vec4(viewPos, 1.0f);

    // Transform device depth into depth range
    float far = gl_DepthRange.far; 
    float near = gl_DepthRange.near;

    float deviceDepth = clipPos.z / clipPos.w;
    gl_FragDepth = (deviceDepth * (far - near) + near + far) * 0.5f;

    float vel = fragInput.velocity;
    fragColor = vec4(vel, vel, vel * 5.0f, 1.0f);
    fragColor.rgb *= getLightColor(normal, fragInput.worldPos);
}
