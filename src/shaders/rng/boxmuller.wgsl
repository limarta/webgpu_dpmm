@group(0) @binding(0) var<uniform> N: u32;
@group(0) @binding(1) var<storage> rng: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> samples: array<f32>;

override nTPB: u32 = 16;
const MAX: u32 = 0xFFFFFFFF;

@compute @workgroup_size(nTPB, 1, 1)
fn boxmuller(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let idx: u32 = global_invocation_id.x;
    if (idx >= N) {
        return;
    }
    let u_0: u32 = rng[idx].x;
    let u_1: u32 = rng[idx].y;

    // Box-Muller transform
    let r: f32 = f32(u_0) / f32(MAX);
    let theta: f32 = f32(u_1) / f32(MAX) * 2.0 * 3.1415926535897932384626433832795;
    let x: f32 = sqrt(-2.0 * log(r)) * cos(theta);
    // let y: f32 = sqrt(-2.0 * log(r)) * sin(theta);

    samples[idx] = x;
}