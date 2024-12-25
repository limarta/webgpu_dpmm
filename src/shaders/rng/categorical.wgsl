@group(0) @binding(0) var<uniform> N: u32; // number of samples
@group(0) @binding(1) var<uniform> K: u32; // number of categories
@group(0) @binding(2) var<storage> rng: array<f32>; // N*K random numbers
@group(0) @binding(3) var<storage> logp: array<f32>; // K log probabilities
@group(0) @binding(4) var<storage, read_write> output: array<u32>;

override nTPB: u32 = 16;

@compute @workgroup_size(nTPB, nTPB)
fn categorical_serial(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let idx:u32 = local_invocation_id.x + workgroup_id.x * nTPB;
    let idk:u32 = local_invocation_id.y + workgroup_id.y * nTPB;

    if (idx >= N || idk >= K) {
        return;
    }

    log(-log(rng[idx]))
}