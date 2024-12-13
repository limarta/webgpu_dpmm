@group(0) @binding(0) var<storage> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> sum: array<f32>;

const n:u32 = 512;
const banks_size:u32 = 32;

var<workgroup> temp: array<f32, 532>;

fn bank_conflict_free_idx(idx: u32) -> u32 {
    return idx + idx / banks_size;
}

@compute @workgroup_size(256)
fn sum_1d(@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    var globalThid: u32 = global_invocation_id.x;
    var thid: u32 = local_invocation_id.x;

    if (thid < (n>>1)) {
        temp[bank_conflict_free_idx(2*thid)] = input[2*globalThid];
        temp[bank_conflict_free_idx(2*thid+1)] = input[2*globalThid+1];
    }
    workgroupBarrier();

    var offset:u32 = 1;
    for (var d = n>>1; d > 0; d >>= 1) {
        if (thid < d)
        {
            var ai:u32 = offset*(2*thid+1)-1;
            var bi:u32 = offset*(2*thid+2)-1;
            temp[bank_conflict_free_idx(bi)] += temp[bank_conflict_free_idx(ai)];
        }
        offset *= 2;
        workgroupBarrier();
    }

    if (thid == 0) {
        sum[workgroup_id.x] = temp[bank_conflict_free_idx(n-1)];
    }
}