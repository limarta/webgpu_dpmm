@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<uniform, read_write> ans: f32;
@group(0) @binding(3) var<uniform, read> N: u32;
const n:u32 = 512;
var<workgroup> temp: array<f32, 532>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    var thid: u32 = local_invocation_id.x;
    var globalThid: u32 = global_invocation_id.x;
    if(thid )
}

fn sum() {
}