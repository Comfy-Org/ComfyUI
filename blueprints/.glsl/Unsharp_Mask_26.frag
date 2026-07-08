#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform float u_float0;  // amount    [0.0 - 3.0]  typical: 0.5-1.5
uniform float u_float1;  // radius    [0.5 - 10.0] blur radius in pixels
uniform float u_float2;  // threshold [0.0 - 0.1]  min difference to sharpen

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

float getLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_image0, 0));
    float radius = max(u_float1, 0.5);
    float amount = u_float0;
    float threshold = u_float2;

    vec4 original = texture(u_image0, v_texCoord);

    // Gaussian blur for the "unsharp" mask
    int samples = int(ceil(radius));
    float sigma = radius / 2.0;

    vec4 blurred = vec4(0.0);
    float totalWeight = 0.0;

    for (int x = -samples; x <= samples; x++) {
        for (int y = -samples; y <= samples; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel;
            vec4 sample_color = texture(u_image0, v_texCoord + offset);

            float dist = length(vec2(float(x), float(y)));
            float weight = gaussian(dist, sigma);
            blurred += sample_color * weight;
            totalWeight += weight;
        }
    }
    blurred /= totalWeight;

    // Unsharp mask = original - blurred
    vec3 mask = original.rgb - blurred.rgb;

    // Luminance-based threshold with smooth falloff
    float lumaDelta = abs(getLuminance(original.rgb) - getLuminance(blurred.rgb));
    float thresholdScale = smoothstep(0.0, threshold, lumaDelta);
    mask *= thresholdScale;

    // Sharpen: original + mask * amount
    vec3 sharpened = original.rgb + mask * amount;

    fragColor0 = vec4(clamp(sharpened, 0.0, 1.0), original.a);
}
