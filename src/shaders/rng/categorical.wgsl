@group(0) @binding(0) var<uniform> dims: vec2u; // number of samples
@group(0) @binding(1) var<storage> rng: array<f32>; // N*K random numbers
@group(0) @binding(2) var<storage> logp: array<f32>; // K log probabilities
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

override nTPB: u32 = 32;

@compute @workgroup_size(nTPB)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let N:u32 = dims.x;
    let K:u32 = dims.y;
    let thid:u32 = global_invocation_id.x;
    let idx:u32 = thid * K;
    if (thid < N) {
        output[thid] = sample(idx, K);
    }
}

fn gumbel(u:f32) -> f32 {
    return -log(-log(u));
}

fn sample(idx: u32, K: u32) -> u32 {
    var max_val: f32 = -10000000.0;
    var max_idx: u32 = 0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let logp_k: f32 = logp[k];
        let a: f32 = gumbel(rng[idx + k]);
        let val: f32 = logp_k + a;
        if (val > max_val) {
            max_val = val;
            max_idx = k;
        }
    }
    return max_idx;

}