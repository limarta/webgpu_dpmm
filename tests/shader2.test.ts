import {expect, test} from 'vitest'
import {beforeAll} from 'vitest'
import {Ops} from '../src/utils/ops'
import {GPUUtils} from '../src/utils/gpu'
import { L } from 'vitest/dist/chunks/reporters.D7Jzd9GS.js';

let device: GPUDevice;
function createPass(shaders) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const shader of shaders) {
        shader.encode(pass);
    }
    pass.end();
    device.queue.submit([encoder.finish()])
}

beforeAll(async () => {
    if (navigator.gpu === undefined) {
        const h = document.querySelector('#title') as HTMLElement;
        h.innerText = 'WebGPU is not supported in this browser.';
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) {
      const h = document.querySelector('#title') as HTMLElement;
      h.innerText = 'No adapter is available for WebGPU.';
      return;
    }
    device = await adapter.requestDevice();
});

test('unsorted_segment_sum_2d_basic', async() => {
    let M = 5;
    let N = 3;
    let data = new Float32Array([
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ])
    let segmentIds = new Uint32Array([0, 1, 0, 1, 0])
    let K = 2;
    let expected = new Float32Array([
        9, 6,
        24, 16,
        39, 26
    ]);

    const inputBuffer = GPUUtils.createStorageBuffer(device, data);
    const segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
    const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));

    const shader = new Ops.UnsortedSegmentSum2DShader(M, N, K);
    await shader.setup(device, inputBuffer, segmentIdsBuffer, outputBuffer);

    createPass([shader]);

    await device.queue.onSubmittedWorkDone();
    const output = await GPUUtils.writeToCPU(device, outputBuffer, N*K*4, false);

    expect(output).toEqual(expected);
})

test('unsorted_segment_sum_2d', async() => {
    let M = 3;
    let N = 1;
    let K = 2;
    const PHI = 1.61803398875;

    function segmentSum2d(data, segmentIds, M, N, K) {
        let output = new Float32Array(N*K);
        for (let j = 0 ; j < N ; j++) {
            for (let i = 0 ; i < M ; i++) {
                let segmentId = segmentIds[i];
                output[j*K+segmentId] += data[i + j * M];
            }
        }
        return output;
    }

    for(let i = 0 ; i < 18 ; i++) {
        M = Math.floor(M * PHI);
        let data = new Float32Array(M*N);
        let segmentIds = new Uint32Array(M);
        for (let j = 0 ; j < M*N ; j++) {
            data[j] = j+1;
        }

        for (let j = 0 ; j < M ; j++) {
            segmentIds[j] = j % K;
        }
        const expected = segmentSum2d(data, segmentIds, M, N, K); 

        let inputBuffer = GPUUtils.createStorageBuffer(device, data);
        let segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));
        let shader = new Ops.UnsortedSegmentSum2DShader(M, N, K);
        await shader.setup(device, inputBuffer, segmentIdsBuffer, outputBuffer);

        createPass([shader]);

        await device.queue.onSubmittedWorkDone();
        let output = await GPUUtils.writeToCPU(device, outputBuffer, N*K*4, false);
        expect(output).toEqual(expected)
    }
});

test('sum3d_basic', async() => {
    let M = 2;
    let N = 3;
    let K = 4;

    let data = new Float32Array([
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24
    ]);

    let expected = new Float32Array([
        3, 7, 11,
        15, 19, 23,
        27, 31, 35,
        39, 43, 47
    ])

    let inputBuffer = GPUUtils.createStorageBuffer(device, data);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));

    let shader = new Ops.Sum3DShader(M, N, K);
    await shader.setup(device, inputBuffer, outputBuffer);
    createPass([shader]);

    await device.queue.onSubmittedWorkDone();
    let output = await GPUUtils.writeToCPU(device, outputBuffer, N*K*4, false);
    expect(output).toEqual(expected);
});

test('sum3d', async() => {
    let M = 3;
    let N = 2;
    let K = 2;
    const PHI = 1.61803398875;

    function sum3d(data, M, N, K) {
        let output = new Float32Array(N*K);
        for (let j = 0 ; j < N ; j++) {
            for (let i = 0 ; i < M ; i++) {
                for (let k = 0 ; k < K ; k++) {
                    output[j + k * N] += data[i + j * M + k * M * N];
                }
            }
        }
        return output;
    }

    for(let i = 0 ; i < 14 ; i++) {
        M = Math.floor(M * PHI);

        let data = new Float32Array(M*N*K);
        for (let j = 0 ; j < M*N*K ; j++) {
            data[j] = j+1;
        }
        let expected = sum3d(data, M, N, K);
        
        let inputBuffer = GPUUtils.createStorageBuffer(device, data);
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));
        let shader = new Ops.Sum3DShader(M, N, K);
        await shader.setup(device, inputBuffer, outputBuffer);
        createPass([shader]);

        await device.queue.onSubmittedWorkDone();
        let output = await GPUUtils.writeToCPU(device, outputBuffer, N*K*4, false);
        expect(output).toEqual(expected)
    }
})

test('shortest_pairwise_loop_basic_0', async() => {
    let input1 = new Float32Array([
        0, 1, 2, 3, 5
    ]);
    
    let input2 = new Float32Array([
        0, -1, 2, 6,
    ]);
    let expected = new Uint32Array([0, 0, 2, 2, 3]);

    let inputBuffer1 = GPUUtils.createStorageBuffer(device, input1);
    let inputBuffer2 = GPUUtils.createStorageBuffer(device, input2);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(5));
    let shader = new Ops.ClosestPairwiseLoopShader(5, 4, 1);

    await shader.setup(device, inputBuffer1, inputBuffer2, outputBuffer);
    createPass([shader]);

    await device.queue.onSubmittedWorkDone();
    let output = await GPUUtils.writeToCPU(device, outputBuffer, 5*4, true);
    expect(output).toEqual(expected);

});

test('shortest_pairwise_loop_basic_1', async() => {
    let input1 = new Float32Array([
        0, 0, 3, // feature 1
        0, 1, 3 // feature 2
    ]);
    
    let input2 = new Float32Array([
        0, 4, 
        0, 4,
    ]);
    let expected = new Uint32Array([0, 0, 1]);

    let inputBuffer1 = GPUUtils.createStorageBuffer(device, input1);
    let inputBuffer2 = GPUUtils.createStorageBuffer(device, input2);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(3));
    let shader = new Ops.ClosestPairwiseLoopShader(3, 2, 2);

    await shader.setup(device, inputBuffer1, inputBuffer2, outputBuffer);
    createPass([shader]);

    await device.queue.onSubmittedWorkDone();
    let output = await GPUUtils.writeToCPU(device, outputBuffer, 3*4, true);
    expect(output).toEqual(expected);

});

test('shortest_pairwise_loop', async() => {
    let M1 = 3;
    let M2 = 5;
    let K = 4;
    const PHI = 1.61803398875;

    for(let i = 0 ; i < 22 ; i++) {
        M1 = Math.floor(M1 * PHI);
        let input1 = new Float32Array(M1*K);
        let input2 = new Float32Array(M2*K);
        for (let j = 0 ; j < M1*K ; j++) {
            input1[j] = (j+1) % 10;
        }
        for (let j = 0 ; j < M2*K ; j++) {
            input2[j] = (j+1) % 7;
        }
        let expected = new Uint32Array(M1);

        for (let j = 0 ; j < M1 ; j++) {
            let minDist = Number.MAX_VALUE;
            let minIndex = 0;
            for (let k = 0 ; k < M2 ; k++) {
                let dist = 0;
                for (let l = 0 ; l < K ; l++) {
                    dist += Math.pow(input1[j + l * M1] - input2[k+l*M2], 2);
                }
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = k;
                }
            }
            expected[j] = minIndex;
        }

        let inputBuffer1 = GPUUtils.createStorageBuffer(device, input1);
        let inputBuffer2 = GPUUtils.createStorageBuffer(device, input2);
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(M1));
        let shader = new Ops.ClosestPairwiseLoopShader(M1, M2, K);

        await shader.setup(device, inputBuffer1, inputBuffer2, outputBuffer);
        createPass([shader]);

        await device.queue.onSubmittedWorkDone();
        let output = await GPUUtils.writeToCPU(device, outputBuffer, M1*4, true);
        expect(output).toEqual(expected);
    }
});

test('count_basic', async() => {
    let data = new Uint32Array([0, 1, 0, 1, 0, 2]);
    
    let inputBufer = GPUUtils.createStorageBuffer(device, data);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(3));
    let shader = new Ops.CountShader(6, 3);

    await shader.setup(device, inputBufer, outputBuffer);
    createPass([shader]);

    await device.queue.onSubmittedWorkDone();
    let output = await GPUUtils.writeToCPU(device, outputBuffer, 3*4, false);
    expect(output).toEqual(new Float32Array([3, 2, 1]));
});

test('count_0', async() => {
    let M = 3;
    let PHI = 1.61803398875;
    let K = 4;

    for(let i = 0 ; i < 20 ; i++) {
        M = Math.floor(M * PHI);
        let data = new Uint32Array(M);
        for (let j = 0 ; j < M ; j++) {
            data[j] = j % K;
        }
        let expected = new Float32Array(K);
        for (let j = 0 ; j < M ; j++) {
            expected[data[j]] += 1;
        }

        let inputBuffer = GPUUtils.createStorageBuffer(device, data);
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
        let shader = new Ops.CountShader(M, K);
        await shader.setup(device, inputBuffer, outputBuffer);
        createPass([shader]);

        await device.queue.onSubmittedWorkDone();
        let output = await GPUUtils.writeToCPU(device, outputBuffer, K*4, false);
        expect(output).toEqual(expected);
    }
});

test('count_1', async() => {
    let M = 3;
    let PHI = 1.61803398875;
    let K = 33;

    for(let i = 0 ; i < 10 ; i++) {
        M = Math.floor(M * PHI);
        let data = new Uint32Array(M);
        for (let j = 0 ; j < M ; j++) {
            data[j] = j % K;
        }
        let expected = new Float32Array(K);
        for (let j = 0 ; j < M ; j++) {
            expected[data[j]] += 1;
        }

        let inputBuffer = GPUUtils.createStorageBuffer(device, data);
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
        let shader = new Ops.CountShader(M, K);
        await shader.setup(device, inputBuffer, outputBuffer);
        createPass([shader]);

        await device.queue.onSubmittedWorkDone();
        let output = await GPUUtils.writeToCPU(device, outputBuffer, K*4, false);
        expect(output).toEqual(expected);
    }
});