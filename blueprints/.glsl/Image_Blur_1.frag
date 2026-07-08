#version 300 es
#pragma passes 2
precision highp float;

// Blur type constants
const int BLUR_GAUSSIAN = 0;
const int BLUR_BOX = 1;
const int BLUR_RADIAL = 2;

// Radial blur config
const int RADIAL_SAMPLES = 12;
const float RADIAL_STRENGTH = 0.0003;

uniform sampler2D u_image0;
uniform int u_int0;      // Blur type (BLUR_GAUSSIAN, BLUR_BOX, BLUR_RADIAL)
uniform float u_float0;  // Blur radius/amount
uniform int u_pass;      // Pass index (0 = horizontal, 1 = vertical)

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(u_image0, 0));
    float radius = max(u_float0, 0.0);

    // Radial (angular) blur - single pass, doesn't use separable
    if (u_int0 == BLUR_RADIAL) {
        // Only execute on first pass
        if (u_pass > 0) {
            fragColor0 = texture(u_image0, v_texCoord);
            return;
        }

        vec2 center = vec2(0.5);
        vec2 dir = v_texCoord - center;
        float dist = length(dir);

        if (dist < 1e-4) {
            fragColor0 = texture(u_image0, v_texCoord);
            return;
        }

        vec4 sum = vec4(0.0);
        float totalWeight = 0.0;
        float angleStep = radius * RADIAL_STRENGTH;

        dir /= dist;

        float cosStep = cos(angleStep);
        float sinStep = sin(angleStep);

        float negAngle = -float(RADIAL_SAMPLES) * angleStep;
        vec2 rotDir = vec2(
            dir.x * cos(negAngle) - dir.y * sin(negAngle),
            dir.x * sin(negAngle) + dir.y * cos(negAngle)
        );

        for (int i = -RADIAL_SAMPLES; i <= RADIAL_SAMPLES; i++) {
            vec2 uv = center + rotDir * dist;
            float w = 1.0 - abs(float(i)) / float(RADIAL_SAMPLES);
            sum += texture(u_image0, uv) * w;
            totalWeight += w;

            rotDir = vec2(
                rotDir.x * cosStep - rotDir.y * sinStep,
                rotDir.x * sinStep + rotDir.y * cosStep
            );
        }

        fragColor0 = sum / max(totalWeight, 0.001);
        return;
    }

    // Separable Gaussian / Box blur
    int samples = int(ceil(radius));

    if (samples == 0) {
        fragColor0 = texture(u_image0, v_texCoord);
        return;
    }

    // Direction: pass 0 = horizontal, pass 1 = vertical
    vec2 dir = (u_pass == 0) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    float sigma = radius / 2.0;

    for (int i = -samples; i <= samples; i++) {
        vec2 offset = dir * float(i) * texelSize;
        vec4 sample_color = texture(u_image0, v_texCoord + offset);

        float weight;
        if (u_int0 == BLUR_GAUSSIAN) {
            weight = gaussian(float(i), sigma);
        } else {
            // BLUR_BOX
            weight = 1.0;
        }

        color += sample_color * weight;
        totalWeight += weight;
    }

    fragColor0 = color / totalWeight;
}
