#version 410

in vec2 v_texcoord;

uniform sampler2D motion_tex;
uniform sampler2D prev_noise_tex;
uniform vec2 u_resolution;
uniform int u_frame;

out vec4 out_noise;

// --- PCG hash + Box-Muller Gaussian PRNG ---

uint pcg_hash(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float uint_to_float(uint x) {
    return float(x) / 4294967296.0;
}

vec2 box_muller(uint seed1, uint seed2) {
    float u1 = max(uint_to_float(pcg_hash(seed1)), 1e-7);
    float u2 = uint_to_float(pcg_hash(seed2));
    float r = sqrt(-2.0 * log(u1));
    float theta = 6.28318530718 * u2;
    return vec2(r * cos(theta), r * sin(theta));
}

vec4 pcg_gaussian(ivec2 pixel, int frame) {
    uint base = uint(pixel.x) * 7919u + uint(pixel.y) * 6271u + uint(frame) * 104729u;
    vec2 g01 = box_muller(base, base + 1u);
    vec2 g23 = box_muller(base + 2u, base + 3u);
    return vec4(g01.x, g01.y, g23.x, g23.y);
}

// --- Nearest-neighbor backward warp ---
// Regaussianization is done CPU-side after readback (requires scatter ops).

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    vec2 mv = texelFetch(motion_tex, pixel, 0).rg;
    vec2 tc = (vec2(pixel) + 0.5) / u_resolution;
    ivec2 src = ivec2(floor((tc - mv) * u_resolution));

    bool oob = any(lessThan(src, ivec2(0)))
            || any(greaterThanEqual(src, ivec2(u_resolution)));

    if (oob) {
        out_noise = pcg_gaussian(pixel, u_frame);
    } else {
        out_noise = texelFetch(prev_noise_tex, src, 0);
    }
}
