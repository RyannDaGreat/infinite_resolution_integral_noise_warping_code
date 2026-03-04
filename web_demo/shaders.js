/**
 * GLSL ES 300 shader sources. Ported from game_demo/shaders/.
 * Only 4 shaders needed: scene pair + display pair.
 * (Noise warp is done CPU-side via particle_warp.js)
 */

export const sceneVert = `#version 300 es
precision highp float;
precision highp int;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat4 u_prev_model;
uniform mat4 u_prev_view_proj;

out vec3 v_color;
out vec4 v_curr_clip;
out vec4 v_prev_clip;
out vec3 v_normal;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_curr_clip = u_view_proj * world_pos;

    vec4 prev_world_pos = u_prev_model * vec4(in_position, 1.0);
    v_prev_clip = u_prev_view_proj * prev_world_pos;

    v_color = in_color;
    v_normal = mat3(u_model) * in_normal;

    gl_Position = v_curr_clip;
}
`;

export const sceneFrag = `#version 300 es
precision highp float;
precision highp int;

in vec3 v_color;
in vec4 v_curr_clip;
in vec4 v_prev_clip;
in vec3 v_normal;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec2 frag_motion;

void main() {
    // Simple directional lighting
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normalize(v_normal), light_dir), 0.3);
    frag_color = vec4(v_color * diff, 1.0);

    // Motion vector: NDC displacement scaled to UV units
    vec2 curr_ndc = v_curr_clip.xy / v_curr_clip.w;
    vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;
    frag_motion = (curr_ndc - prev_ndc) * 0.5;
}
`;

export const displayVert = `#version 300 es
precision highp float;
precision highp int;

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    v_texcoord = in_texcoord;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
`;

export const displayFrag = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_texcoord;

uniform sampler2D noise_tex;
uniform sampler2D color_tex;
uniform sampler2D motion_tex;
uniform int u_display_mode; // 0=noise, 1=color, 2=motion, 3=side-by-side, 4=raw

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
`;
