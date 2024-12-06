@group(0) @binding(0) var<storage, read_write> seed : array<vec4u>;
@group(0) @binding(1) var<storage, read_write> random : array<f32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(local_invocation_id) local_invocation_id: vec3u) {
    var a = tau_step(seed[workgroup_id.x].x, 13, 19, 12, 4294967294);
    var b = tau_step(seed[workgroup_id.x].y, 2, 25, 4, 4294967288);
    var c = tau_step(seed[workgroup_id.x].z, 3, 11, 17, 4294967280);
    var d = lcg_step(seed[workgroup_id.x].w);

    var r = f32(a ^ b ^ c ^ d) / 4294967296.0;
    random[workgroup_id.x] = r;
}

fn tau_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    var b = u32((((z << s1) ^ z) >> s2));
    return (((z & m)) << s3) ^ b;
}

fn lcg_step(state: u32) -> u32 {
    return (state * 1664525 + 1013904223);
}