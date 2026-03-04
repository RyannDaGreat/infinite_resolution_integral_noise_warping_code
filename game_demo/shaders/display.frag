#version 410

in vec2 v_texcoord;

uniform sampler2D noise_tex;
uniform sampler2D color_tex;
uniform sampler2D motion_tex;
uniform int u_display_mode;  // 0=noise, 1=color, 2=motion, 3=side-by-side, 4=raw image

out vec4 frag_color;

void main() {
    if (u_display_mode == 0) {
        vec4 noise = texture(noise_tex, v_texcoord);
        frag_color = vec4(noise.rgb / 5.0 + 0.5, 1.0);
    } else if (u_display_mode == 1) {
        frag_color = texture(color_tex, v_texcoord);
    } else if (u_display_mode == 2) {
        vec2 mv = texture(motion_tex, v_texcoord).rg;
        frag_color = vec4(mv * 5.0 + 0.5, 0.5, 1.0);
    } else if (u_display_mode == 3) {
        // Side by side: left=color, right=noise
        if (v_texcoord.x < 0.5) {
            vec2 uv = vec2(v_texcoord.x * 2.0, v_texcoord.y);
            frag_color = texture(color_tex, uv);
        } else {
            vec2 uv = vec2((v_texcoord.x - 0.5) * 2.0, v_texcoord.y);
            vec4 noise = texture(noise_tex, uv);
            frag_color = vec4(noise.rgb / 5.0 + 0.5, 1.0);
        }
    } else {
        // Raw: noise_tex.rgb displayed directly (for image warp mode)
        frag_color = vec4(texture(noise_tex, v_texcoord).rgb, 1.0);
    }
}
