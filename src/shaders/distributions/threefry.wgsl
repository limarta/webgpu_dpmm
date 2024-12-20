// https://dl.acm.org/doi/10.1145/2063384.2063405
@group(0) @binding(0) var<uniform> key: vec4<u32>;
@group(0) @binding(1) var<uniform> N: u32;
@group(0) @binding(2) var<storage,read_write> rng: array<vec4u>;

override nTPB:u32 = 16;
const N_w: u32 = 4;
const C20: u32 = 0xA9FC1A22;
const R: u32 = 12;
@compute @workgroup_size(nTPB,1,1)
fn threefry(
    @builtin(global_invocation_id) global_invocation_id:vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
){
    let id = global_invocation_id.x;
    if (id < N) {
        let K_nw = C20 ^ key[0] ^ key[1] ^ key[2] ^ key[3];
        var current_key: vec4<u32> = key;
        var v_di: vec4<u32> = vec4(0,0,0,id+1);
        for(var d:u32 = 0 ; d < R ; d++) {
            var e_di:vec4u = vec4(0,0,0,0);
            if(d % 4 == 0) {
                current_key = subkey(key, K_nw, d >> 2);
                e_di = current_key + v_di;
            } else {
                e_di = v_di;
            }

            let f_di = mix(e_di, d);

            v_di = f_di.xwzy;
        }
        rng[id] = v_di;
    }
}

fn subkey(key: vec4<u32>, K_nw: u32, s: u32) -> vec4u {
    let k_1: u32 = subkey_(key, K_nw, s, 0);
    let k_2: u32 = subkey_(key, K_nw, s, 1);
    let k_3: u32 = subkey_(key, K_nw, s, 2);
    let k_4: u32 = subkey_(key, K_nw, s, 3) + s;
    return vec4(k_1, k_2, k_3, k_4);
}

fn subkey_(key: vec4<u32>, K_nw: u32, s:u32, i:u32) -> u32 {
    let idx: u32 = (s + i) % (N_w+1);
    switch idx {
        case 0,1,2,3: {
            return key[idx];
        } 
        case 4: {
            return K_nw;
        }
        default: {
            return 0;
        }
    }
}

fn mix(x: vec4<u32>, d: u32) -> vec4<u32> {
    let y_0 = x[0] + x[1];
    let z_0 = x[2] + x[3];

    let y_1 = rotatebits(x[1], rotation_shift(d,0)) ^ y_0; // Change to rotation 
    let z_1 = rotatebits(x[3], rotation_shift(d,1)) ^ z_0; // Change to rotation 
    return vec4(y_0, y_1, z_0, z_1);
}

fn rotatebits(x:u32, d:u32) -> u32 {
    return (x << d) | (x >> (32-d));
}

fn rotation_shift(d:u32,j:u32) -> u32 {
    let idx:u32 = 2*(d & 7) + j;
    switch idx {
        case 0: {
            return 10;
        }
        case 1: {
            return 26;
        }
        case 2: {
            return 11;
        }
        case 3: {
            return 21;
        }
        case 4: {
            return 13;
        }
        case 5: {
            return 27;
        }
        case 6: {
            return 23;
        }
        case 7: {
            return 5;
        }
        case 8: {
            return 6;
        }
        case 9: {
            return 20;
        }
        case 10: {
            return 17;
        }
        case 11: {
            return 11;
        }
        case 12: {
            return 25;
        }
        case 13: {
            return 10;
        }
        case 14: {
            return 18;
        }
        case 15: {
            return 20;
        }
        default: {
            return 0;
        }
    }
}

