@group(0) @binding(0) var<uniform> dim:u32
@group(0) @binding(1) var<storage> data:array<f32>
@group(0) @binding(2) var<storage> indices:array<u32>
@group(0) @binding(3) var<storage-read_write> output:array<f32>

override nTPB:u32 = 32;

@compute @workgroup_size(nTPB, 1,1)
fn index(
    @builtin(global_invocation_id) global_invocation_id:vec3<u32>
) {
    let thid:u32 = global_invocation_id.x;
    if(thid >= dim) {
        return;
    }

    output[thid] = data[indices[thid]];
}