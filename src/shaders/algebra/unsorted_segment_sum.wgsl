@group(0) @binding(0) var<storage> data: array<f32>;
@group(0) @binding(1) var<storage> segments: array<u32>;
@group(0) @binding(2) var<uniform> num_segments: u32;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const n:u32 = 32;
var<workgroup> temp: array<f32, 32>;

@compute @workgroup_size(16,1,1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    var thid:u32 = local_invocation_id.x;
    var gThid:u32 = workgroup_id.x*(n>>1) + thid;

    if (thid < (n>>1)) {
        var left:f32 = select(0.0, data[2*gThid], segments[2*gThid] == workgroup_id.y);
        var right:f32 = select(0.0, data[2*gThid+1], segments[2*gThid+1] == workgroup_id.y);
        temp[thid] = left + right;
    }
    workgroupBarrier();

    var offset:u32 = 1;

    for(var d = n>>1; d > 0 ; d>>=1) {
        if (thid < d) {
            var ai:u32 = offset*(2*thid+1)-1;
            var bi:u32 = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        workgroupBarrier();
    }

    let stride:u32 = num_workgroups.x;
    if (thid == 0) {
        output[workgroup_id.y*stride+workgroup_id.x] = temp[n-1];
    }
}