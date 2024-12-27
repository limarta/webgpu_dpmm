@group(0) @binding(0) var<uniform> arr_dims: vec2<u32>;
@group(0) @binding(1) var<storage> arr: array<f32>;
@group(0) @binding(2) var<storage> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

override nTPB: u32 = 32;
override op: u32 = 0; // 0: add, 1: sub, 2: mul, 3: div

@compute @workgroup_size(nTPB, 1, 1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
) {
    let idx: u32 =  global_invocation_id.x;
    let idy: u32 =  idx / arr_dims.y;
    if (idx >= arr_dims.x * arr_dims.y) {
        return;
    }
    switch (op) {
        case 0: {
            output[idx] = arr[idx] + v[idy];
        }
        case 1: {
            output[idx] = arr[idx] - v[idy];
        }
        case 2: {
            output[idx] = arr[idx] * v[idy];
        }
        case 3: {
            output[idx] = arr[idx] / v[idy];
        }
        default: {
            output[idx] = arr[idx] + v[idy];
        }
    }
}