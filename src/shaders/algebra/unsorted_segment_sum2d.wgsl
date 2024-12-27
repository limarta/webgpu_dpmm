@group(0) @binding(0) var<uniform> dims: vec2<u32>;
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
    let DIMX:u32 = dims.x;
    let DIMY:u32 = dims.y;

    let GRID_STRIDE: u32 = num_workgroups.x * nTPB;
    let MAX_BLOCKS_X:u32 = (DIMX+nTPB-1) / nTPB;

    var thid:u32 = local_invocation_id.x;
    var idx:u32 = thid + workgroup_id.x * nTPB;
    var idy:u32 =  workgroup_id.y;
    var idk:u32 = workgroup_id.z;

    var val:f32 = 0.0;
    while (idx < DIMX) {
        var gThid:u32 = idx + idy * DIMX;
        var gThid2:u32 = idx;
        val += select(0.0, data[gThid], segments_ids[gThid2] == idk);
        idx += GRID_STRIDE;
    }
    temp[thid] = val;

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

    if (thid == 0) {
        let index = workgroup_id.x + idy * MAX_BLOCKS_X + idk * MAX_BLOCKS_X * DIMY;
        output[index] = temp[nTPB-1];
    }
}