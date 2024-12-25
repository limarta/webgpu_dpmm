@group(0) @binding(0) var<uniform> dims: vec2<u32>;
@group(0) @binding(1) var<storage> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

override nTPB:u32 = 32;
override TMP_LEN:u32 = nTPB;
var<workgroup> temp: array<f32, TMP_LEN>;

@compute @workgroup_size(nTPB, 1, 1)
fn sum_2d_within_block(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
){
    let GRID_STRIDE: u32 = num_workgroups.x * nTPB;
    let COLS: u32 = dims.x;
    let ROWS: u32 = dims.y;
    let MAX_BLOCKS_X:u32 = (COLS+TMP_LEN-1) / TMP_LEN;
    
    var idx: u32 = local_invocation_id.x + workgroup_id.x * nTPB;
    let idy: u32 = workgroup_id.y;
    let thid: u32 = local_invocation_id.x;

    var val:f32 = 0;
    while (idx < COLS) {
        val += input[idy*COLS + idx];
        idx += GRID_STRIDE;
    }
    temp[thid] = val;

    workgroupBarrier();

    var offset:u32 = 1;
    for (var d = TMP_LEN>>1; d > 0; d >>= 1) {
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
        var idx = idy * MAX_BLOCKS_X + workgroup_id.x;
        output[idx] = temp[TMP_LEN-1];
    }
}

@compute @workgroup_size(nTPB,1,1)
fn sum_2d_final(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let GRID_STRIDE: u32 = nTPB;
    let COLS: u32 = dims.x;
    let ROWS: u32 = dims.y;

    var idx: u32 = local_invocation_id.x;
    let idy: u32 = workgroup_id.y;

    let thid: u32 = local_invocation_id.x;
    var val:f32 = 0;
    while (idx < COLS) {
        val += input[idy*COLS + idx];
        idx += GRID_STRIDE;
    }
    temp[thid] = val;

    workgroupBarrier();

    var offset:u32 = 1;
    for (var d = TMP_LEN>>1; d > 0; d >>= 1) {
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
        // sum[idy] = temp[TMP_LEN-1];
    }
}