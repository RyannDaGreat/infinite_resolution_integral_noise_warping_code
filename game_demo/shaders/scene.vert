#version 410

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
