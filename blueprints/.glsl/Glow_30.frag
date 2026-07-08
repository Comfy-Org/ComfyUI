#version 300 es
precision mediump float;

uniform sampler2D u_image0;
uniform int u_int0;      // Blend mode
uniform int u_int1;      // Color tint
uniform float u_float0;  // Intensity
uniform float u_float1;  // Radius
uniform float u_float2;  // Threshold

in vec2 v_texCoord;
out vec4 fragColor;

const int BLEND_ADD      = 0;
const int BLEND_SCREEN   = 1;
const int BLEND_SOFT     = 2;
const int BLEND_OVERLAY  = 3;
const int BLEND_LIGHTEN  = 4;

const float GOLDEN_ANGLE = 2.39996323;
const int MAX_SAMPLES = 48;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 hexToRgb(int h) {
    return vec3(
        float((h >> 16) & 255),
        float((h >> 8) & 255),
        float(h & 255)
    ) * (1.0 / 255.0);
}

vec3 blend(vec3 base, vec3 glow, int mode) {
    if (mode == BLEND_SCREEN) {
        return 1.0 - (1.0 - base) * (1.0 - glow);
    }
    if (mode == BLEND_SOFT) {
        return mix(
            base - (1.0 - 2.0 * glow) * base * (1.0 - base),
            base + (2.0 * glow - 1.0) * (sqrt(base) - base),
            step(0.5, glow)
        );
    }
    if (mode == BLEND_OVERLAY) {
        return mix(
            2.0 * base * glow,
            1.0 - 2.0 * (1.0 - base) * (1.0 - glow),
            step(0.5, base)
        );
    }
    if (mode == BLEND_LIGHTEN) {
        return max(base, glow);
    }
    return base + glow;
}

void main() {
    vec4 original = texture(u_image0, v_texCoord);
    
    float intensity = u_float0 * 0.05;
    float radius = u_float1 * u_float1 * 0.012;
    
    if (intensity < 0.001 || radius < 0.1) {
        fragColor = original;
        return;
    }
    
    float threshold = 1.0 - u_float2 * 0.01;
    float t0 = threshold - 0.15;
    float t1 = threshold + 0.15;
    
    vec2 texelSize = 1.0 / vec2(textureSize(u_image0, 0));
    float radius2 = radius * radius;
    
    float sampleScale = clamp(radius * 0.75, 0.35, 1.0);
    int samples = int(float(MAX_SAMPLES) * sampleScale);
    
    float noise = hash(gl_FragCoord.xy);
    float angleOffset = noise * GOLDEN_ANGLE;
    float radiusJitter = 0.85 + noise * 0.3;
    
    float ca = cos(GOLDEN_ANGLE);
    float sa = sin(GOLDEN_ANGLE);
    vec2 dir = vec2(cos(angleOffset), sin(angleOffset));
    
    vec3 glow = vec3(0.0);
    float totalWeight = 0.0;
    
    // Center tap
    float centerMask = smoothstep(t0, t1, dot(original.rgb, LUMA));
    glow += original.rgb * centerMask * 2.0;
    totalWeight += 2.0;
    
    for (int i = 1; i < MAX_SAMPLES; i++) {
        if (i >= samples) break;
        
        float fi = float(i);
        float dist = sqrt(fi / float(samples)) * radius * radiusJitter;
        
        vec2 offset = dir * dist * texelSize;
        vec3 c = texture(u_image0, v_texCoord + offset).rgb;
        float mask = smoothstep(t0, t1, dot(c, LUMA));
        
        float w = 1.0 - (dist * dist) / (radius2 * 1.5);
        w = max(w, 0.0);
        w *= w;
        
        glow += c * mask * w;
        totalWeight += w;
        
        dir = vec2(
            dir.x * ca - dir.y * sa,
            dir.x * sa + dir.y * ca
        );
    }
    
    glow *= intensity / max(totalWeight, 0.001);
    
    if (u_int1 > 0) {
        glow *= hexToRgb(u_int1);
    }
    
    vec3 result = blend(original.rgb, glow, u_int0);
    result += (noise - 0.5) * (1.0 / 255.0);
    
    fragColor = vec4(clamp(result, 0.0, 1.0), original.a);
}