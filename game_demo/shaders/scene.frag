#version 410

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
