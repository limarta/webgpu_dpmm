@group(0) @binding(0) var<uniform> index:u32;
@group(0) @binding(1) var<storage> rng:array<vec4u>;
@group(0) @binding(2) var<storage,read_write> key:vec4u;

@compute @workgroup_size(1,1,1)
fn main() {
    key = rng[index];
}