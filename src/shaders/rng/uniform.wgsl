@group(0) @binding(0) var<uniform> N: u32;
@group(0) @binding(1) var<storage> rng: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> samples: array<vec4f>;

override nTPB: u32 = 16;
const MAX: u32 = 0xFFFFFFFF;

@compute @workgroup_size(nTPB, 1, 1)
fn uni(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let idx: u32 = global_invocation_id.x;
    if (idx >= N) {
        return;
    }
    samples[idx] = vec4f(rng[idx]) / f32(MAX);
}