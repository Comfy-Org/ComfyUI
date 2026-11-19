#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform float u_float0;  // strength [0.0 – 2.0] typical: 0.3–1.0

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_image0, 0));
    
    // Sample center and neighbors
    vec4 center = texture(u_image0, v_texCoord);
    vec4 top    = texture(u_image0, v_texCoord + vec2( 0.0, -texel.y));
    vec4 bottom = texture(u_image0, v_texCoord + vec2( 0.0,  texel.y));
    vec4 left   = texture(u_image0, v_texCoord + vec2(-texel.x,  0.0));
    vec4 right  = texture(u_image0, v_texCoord + vec2( texel.x,  0.0));
    
    // Edge enhancement (Laplacian)
    vec4 edges = center * 4.0 - top - bottom - left - right;
    
    // Add edges back scaled by strength
    vec4 sharpened = center + edges * u_float0;
    
    fragColor0 = vec4(clamp(sharpened.rgb, 0.0, 1.0), center.a);
}