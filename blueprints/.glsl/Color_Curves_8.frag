#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform sampler2D u_curve0;  // RGB master curve (256x1 LUT)
uniform sampler2D u_curve1;  // Red channel curve
uniform sampler2D u_curve2;  // Green channel curve
uniform sampler2D u_curve3;  // Blue channel curve

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

// GIMP-compatible curve lookup with manual linear interpolation.
// Matches gimp_curve_map_value_inline() from gimpcurve-map.c:
//   index = value * (n_samples - 1)
//   f = fract(index)
//   result = (1-f) * samples[floor] + f * samples[ceil]
//
// Uses texelFetch (NEAREST) to avoid GPU half-texel offset issues
// that occur with texture() + GL_LINEAR on small 256x1 LUTs.
float applyCurve(sampler2D curve, float value) {
    value = clamp(value, 0.0, 1.0);

    float pos = value * 255.0;
    int lo = int(floor(pos));
    int hi = min(lo + 1, 255);
    float f = pos - float(lo);

    float a = texelFetch(curve, ivec2(lo, 0), 0).r;
    float b = texelFetch(curve, ivec2(hi, 0), 0).r;

    return a + f * (b - a);
}

void main() {
    vec4 color = texture(u_image0, v_texCoord);

    // GIMP order: per-channel curves first, then RGB master curve.
    // See gimp_curve_map_pixels() default case in gimpcurve-map.c:
    //   dest = colors_curve( channel_curve( src ) )
    float tmp_r = applyCurve(u_curve1, color.r);
    float tmp_g = applyCurve(u_curve2, color.g);
    float tmp_b = applyCurve(u_curve3, color.b);
    color.r = applyCurve(u_curve0, tmp_r);
    color.g = applyCurve(u_curve0, tmp_g);
    color.b = applyCurve(u_curve0, tmp_b);

    fragColor0 = vec4(color.rgb, color.a);
}
