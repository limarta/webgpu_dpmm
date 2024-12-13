@group(0) @binding(0) var<storage, read_write> seed : array<vec4u>;
@group(0) @binding(1) var<storage, read_write> random : array<f32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(local_invocation_id) local_invocation_id: vec3u) {
    // box_muller(workgroup_id.x);
    let idx= 16 * workgroup_id.x + local_invocation_id.x;
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    seed[idx] = pcg4d(seed[idx]);
    // random[idx] = f32(seed[idx].x) / 4294967296.0;
    random[idx] = f32(box_muller(seed[idx]));
}

// fn gumbel(v: vec4u) -> u32 {

// }

fn box_muller_4(v: vec4u) -> vec4f {
    let v1 = pcg4d(v);
    let x = f32(v1.x) / 4294967296.0;
    let y = f32(v1.y) / 4294967296.0;
    let r = sqrt(-2.0 * log(x));
    let theta = 2.0 * 3.14159265359 * y;
    return vec4f(r * cos(theta), r * sin(theta), r * cos(theta), r * sin(theta));
}

fn box_muller(v: vec4u) -> f32{
    let v1 = pcg4d(v);
    let x = f32(v1.x) / 4294967296.0;
    let y = f32(v1.y) / 4294967296.0;
    let r = sqrt(-2.0 * log(x));
    let theta = 2.0 * 3.14159265359 * y;
    return r * cos(theta);
}

fn pcg4d(v: vec4u) -> vec4u {
    var w = v * 1664525u + 1013904223u;
    w.x += w.y * w.z; w.y += w.z * w.x; w.z += w.x * w.y; w.w += w.y * w.z;
    w.x ^= w.x >> 16u;
    w.y ^= w.y >> 16u;
    w.z ^= w.z >> 16u;
    w.w ^= w.w >> 16u;
    w.x += w.y * w.w; w.y += w.z * w.x; w.z += w.x * w.y; w.w += w.y * w.z;
    return w;
}
