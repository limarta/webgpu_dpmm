@group(0) @binding(0) var<uniform> dims: u32;
@group(0) @binding(1) var<storage> rng: array<f32>;
@group(0) @binding(2) var<storage> means: array<f32>;
@group(0) @binding(3) var<storage> std: array<f32>;
@group(0) @binding(4) var<storage> assignments: array<f32>;
@group(0) @binding(5) var<storage> output: array<f32>;

override nTPB: u32 = 32;

@compute @workgroup_size(nTPB, 1, 1)
fn main() {

}