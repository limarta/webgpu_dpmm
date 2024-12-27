@group(0) @binding(0) var<uniform> length: u32;
@group(0) @binding(1) var<storage> data: array<f32>;
@group(0) @binding(2) var<storage> segments_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

override nTPB:u32 = 32;
var<workgroup> temp: array<f32, nTPB>;

@compute @workgroup_size(nTPB,1,1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    var thid:u32 = local_invocation_id.x;
    var idx:u32 = thid + workgroup_id.x * nTPB;
    if (idx < length) {
        var val:f32 = select(0.0, data[idx], segments_ids[idx] == workgroup_id.y);
        temp[thid] = val;
    }

    workgroupBarrier();

    var offset:u32 = 1;

    for(var d = nTPB>>1; d > 0 ; d>>=1) {
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
        output[workgroup_id.y*stride+workgroup_id.x] = temp[nTPB-1];
    }
}