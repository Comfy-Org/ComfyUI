#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform float u_float0;
uniform float u_float1;
uniform float u_float2;
uniform float u_float3;
uniform float u_float4;
uniform float u_float5;
uniform float u_float6;
uniform float u_float7;
uniform float u_float8;
uniform bool u_bool0;

in vec2 v_texCoord;
out vec4 fragColor;

vec3 rgb2hsl(vec3 c) {
    float maxC = max(c.r, max(c.g, c.b));
    float minC = min(c.r, min(c.g, c.b));
    float l = (maxC + minC) * 0.5;
    if (maxC == minC) return vec3(0.0, 0.0, l);
    float d = maxC - minC;
    float s = l > 0.5 ? d / (2.0 - maxC - minC) : d / (maxC + minC);
    float h;
    if (maxC == c.r) {
        h = (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0);
    } else if (maxC == c.g) {
        h = (c.b - c.r) / d + 2.0;
    } else {
        h = (c.r - c.g) / d + 4.0;
    }
    h /= 6.0;
    return vec3(h, s, l);
}

float hue2rgb(float p, float q, float t) {
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0 / 2.0) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    return p;
}

vec3 hsl2rgb(vec3 hsl) {
    float h = hsl.x, s = hsl.y, l = hsl.z;
    if (s == 0.0) return vec3(l);
    float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
    float p = 2.0 * l - q;
    return vec3(
        hue2rgb(p, q, h + 1.0 / 3.0),
        hue2rgb(p, q, h),
        hue2rgb(p, q, h - 1.0 / 3.0)
    );
}

void main() {
    vec4 tex = texture(u_image0, v_texCoord);
    vec3 color = tex.rgb;

    vec3 shadows = vec3(u_float0, u_float1, u_float2) * 0.01;
    vec3 midtones = vec3(u_float3, u_float4, u_float5) * 0.01;
    vec3 highlights = vec3(u_float6, u_float7, u_float8) * 0.01;

    float maxC = max(color.r, max(color.g, color.b));
    float minC = min(color.r, min(color.g, color.b));
    float lightness = (maxC + minC) * 0.5;

    // GIMP weight curves: linear ramps with constants a=0.25, b=0.333, scale=0.7
    const float a = 0.25;
    const float b = 0.333;
    const float scale = 0.7;

    float sw = clamp((lightness - b) / -a + 0.5, 0.0, 1.0) * scale;
    float mw = clamp((lightness - b) / a + 0.5, 0.0, 1.0) *
               clamp((lightness + b - 1.0) / -a + 0.5, 0.0, 1.0) * scale;
    float hw = clamp((lightness + b - 1.0) / a + 0.5, 0.0, 1.0) * scale;

    color += sw * shadows + mw * midtones + hw * highlights;

    if (u_bool0) {
        vec3 hsl = rgb2hsl(clamp(color, 0.0, 1.0));
        hsl.z = lightness;
        color = hsl2rgb(hsl);
    }

    fragColor = vec4(clamp(color, 0.0, 1.0), tex.a);
}
