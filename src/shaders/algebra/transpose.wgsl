@group(0) @binding(0) var<uniform> dims: vec2<u32>;
@group(0) @binding(1) var<storage> arr: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

override nTPB: u32 = 32;
@compute @workgroup_size(nTPB, nTPB, 1)
fn transpose(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let DIMX = dims.x;
    let DIMY = dims.y;

    let idx = workgroup_id.x * nTPB + local_invocation_id.x;
    let idy = workgroup_id.y * nTPB + local_invocation_id.y;

    if (idx >= DIMX || idy >= DIMY) {
        return;
    }

    output[idy * DIMX + idx] = arr[idx * DIMY + idy];
    // output[idy * DIMX + idx] = arr[idx * DIMY + idy];
}