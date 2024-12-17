@group(0) @binding(0) var<uniform> dims: vec2<u32>;
@group(0) @binding(1) var<uniform> axis: u32;
@group(0) @binding(2) var<storage> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

override nTPB:u32 = 16;
override TMP_LEN:u32 = 32;
var<workgroup> temp: array<f32, TMP_LEN>;

@compute @workgroup_size(nTPB, 1, 1)
fn sum_2d_within_block(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
){
    var N_new = u32(ceil(f32(dims.y)/f32(TMP_LEN)));
    var thid: vec2<u32> = vec2<u32>(local_invocation_id.xy);
    var gThid: vec2<u32> = vec2<u32>(global_invocation_id.xy);

    var row = workgroup_id.y;
    if (row < dims.x) {
        var col = 32 * workgroup_id.x + 2 * thid.x;
        if (col < dims.y) {
            temp[2*thid.x] = input[row * dims.y + col];
        }

        col = 32 * workgroup_id.x + 2 * thid.x+1;
        if (col < dims.y && row < dims.x) {
            temp[2*thid.x+1] = input[row * dims.y + col];
        }

    }
    workgroupBarrier();

    var offset:u32 = 1;
    for (var d = TMP_LEN>>1; d > 0; d >>= 1) {
        if (thid.x < d)
        {
            var ai:u32 = offset*(2*thid.x+1)-1;
            var bi:u32 = offset*(2*thid.x+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        workgroupBarrier();
    }

    if (thid.x == 0) {
        var row = workgroup_id.y;
        if (row < dims.x) {
            var idx = row * N_new + workgroup_id.x;
            output[idx] = temp[TMP_LEN-1];
        }
    }
}

@compute @workgroup_size(1,1,1)
fn sum_2d_final() {

}