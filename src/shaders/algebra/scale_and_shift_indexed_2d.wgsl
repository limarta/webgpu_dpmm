@group(0) @binding(0) var<uniform> dims: vec4<u32>;
@group(0) @binding(1) var<storage> assignments: array<u32>;
@group(0) @binding(2) var<storage> scale: array<f32>;
@group(0) @binding(3) var<storage> shift: array<f32>;
@group(0) @binding(4) var<storage,read_write> arr: array<f32>;

override nTPB:u32 = 32u;
@compute @workgroup_size(nTPB,1,1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let M1:u32 = dims.x;
    let N1:u32 = dims.y;
    let M2:u32 = dims.z;
    let N2:u32 = dims.w; 

    let idx: u32 = local_invocation_id.x + nTPB * workgroup_id.x; 
    let idy: u32 = workgroup_id.y;

    if (idx >= M1) {
        return;
    }

    let label: u32 = assignments[idx];
    let index: u32 = label + N1 * idy;
    arr[idx + M1 * idy] = scale[index] * arr[idx + M1 * idy] + shift[index];
}