// "A simple method for generating gamma variables"
// G. Marsaglia and W.W. Tsang
// ACM Transactions on Mathematical Software (TOMS), 2000, Volume 26(3), pp. 363-372
// doi:10.1145/358407.358414
// http://www.cparity.com/projects/AcmClassification/samples/358414.pdf

@group(0) @binding(0) var<uniform> dims: u32;
@group(0) @binding(1) var<uniform> parameters: vec2f;
@group(0) @binding(2) var<storage> uniformRng: array<f32>;
@group(0) @binding(3) var<storage> normalRng: array<f32>;
@group(0) @binding(4) var<storage, read_write> changed: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

override nTPB: u32 = 32;
@compute @workgroup_size(nTPB,1,1)
fn main(
    @builtin(global_invocation_id) globalInvocationId: vec3<u32>,
) {
    let shape = parameters.x;
    let scale = parameters.y;
    let d = shape - 1/3;
    let c = 1 / (3 * sqrt(d));
    let k = d * scale;

    let N = dims;
    let idx = globalInvocationId.x;

    for(var i:u32 = 0 ; i < 4 ; i++) {
        let x:f32 = normalRng[idx+i*N];

        let cbrt_v = 1 + c * x;
        let v = cbrt_v * cbrt_v * cbrt_v;

        let u = uniformRng[idx + i*N];

        // // if u < 1 - 0.0331 * x*x*x*x || log(u) < 0.5 * xsq + d * (1 - v + log(v)) {
        if u < 1 - 0.0331 * x * x * x * x {
            output[idx] = k * v;
            changed[idx] = i;
            return;
        }

    }
}