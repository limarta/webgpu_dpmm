@group(0) @binding(0) var<uniform> dims:vec3u;
@group(0) @binding(1) var<storage> arr1:array<f32>;
@group(0) @binding(2) var<storage> arr2:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<u32>;

override nTPBx:u32 = 32;
override nTPBy:u32 = 1;
override nTPBz:u32 = 1;

var<workgroup> best_dist:array<f32,nTPBy*nTPBz>;

struct Pair {
    index:u32,
    value:f32,
}

@compute @workgroup_size(nTPBx, nTPBy, nTPBz)
fn closest_pairwise_loop(
    @builtin(local_invocation_id) local_invocation_id:vec3<u32>,
    @builtin(workgroup_id) workgroup_id:vec3<u32>,
) {
    let M_1:u32 = dims.x;
    let M_2:u32 = dims.y;
    let K:u32 = dims.z;

    let p_x:u32 = local_invocation_id.x + workgroup_id.x * nTPBx;

    if p_x >= M_1 {
        return;
    }

    let pair = closest_point(p_x);
    output[p_x] = pair.index;
}

fn closest_point(p_x:u32) -> Pair {
    let M:u32 = dims.x;
    let N:u32 = dims.y;
    let K:u32 = dims.z;

    var best:Pair = Pair(0, 0.0);
    var best_dist:f32 = 1e9;
    for (var p_y:u32=0; p_y < N; p_y++) {
        let dist:f32 = distance(p_x, p_y, K);
        if dist < best_dist {
            best_dist = dist;
            best.index = p_y;
            best.value = dist;
        }
    }
    return best;
}

// p_x and p_y are the indices of the two points. K is the number of features.
fn distance(p_x:u32, p_y:u32, K: u32) -> f32 {
    var dist:f32 = 0.0;
    for(var i:u32=0; i < K; i++){
        let diff = arr1[i + p_x * K] - arr2[i + p_y * K];
        dist = dist + diff * diff;
    }
    return dist;

}