@group(0) @binding(0) var<uniform> dims: vec3<u32>;
@group(0) @binding(1) var<storage> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

override nTPB:u32 = 4;
var<workgroup> temp: array<f32, nTPB>;

@compute @workgroup_size(nTPB, 1, 1)
fn sum3d(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
){
    let COLS: u32 = dims.x;
    let ROWS: u32 = dims.y;
    let DEPTH: u32 = dims.z;

    let GRID_STRIDE: u32 = num_workgroups.x * nTPB;
    let MAX_BLOCKS_X:u32 = (COLS+nTPB-1) / nTPB;
    
    let thid: u32 = local_invocation_id.x;
    var idx: u32 = local_invocation_id.x + workgroup_id.x * nTPB;
    let idy: u32 = workgroup_id.y;
    let idz: u32 = workgroup_id.z;

    var val:f32 = 0;
    while (idx < COLS) {
        val += input[idx + idy*COLS + idz * COLS * ROWS] ;
        idx += GRID_STRIDE;
    }
    temp[thid] = val;

    workgroupBarrier();

    var offset:u32 = 1;
    for (var d = nTPB>>1; d > 0; d >>= 1) {
        if (thid < d)
        {
            var ai:u32 = offset*(2*thid+1)-1;
            var bi:u32 = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        workgroupBarrier();
    }

    if (thid == 0) {
        var index = workgroup_id.x + idy * MAX_BLOCKS_X + idz * MAX_BLOCKS_X * ROWS;
        output[index] = temp[nTPB-1];
    }
}