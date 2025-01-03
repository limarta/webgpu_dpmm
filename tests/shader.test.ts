import {expect, test} from 'vitest'
import {beforeAll} from 'vitest'
import {Ops} from '../src/utils/ops'
import {GPUUtils} from '../src/utils/gpu'

let device: GPUDevice;

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

function createPass(shaders) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const shader of shaders) {
        shader.encode(pass);
    }
    pass.end();
    device.queue.submit([encoder.finish()])
}

function expectArraysToBeCloseTo(expected: Float32Array, actual: Float32Array, tolerance: number) {
    expect(expected.length).toBe(actual.length);
    for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(expected[i] - actual[i])).toBeLessThanOrEqual(tolerance);
    }
}

test('transpose_basic', async () => {
    let data = new Float32Array([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    ])
    let expected = new Float32Array([
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12
    ])

    const inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
    const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(12));
    const transposeShader = new Ops.TransposeShader(3,4);

    await transposeShader.setup(device, inputBuffer, outputBuffer);

    createPass([transposeShader]);
    const output = await GPUUtils.writeToCPU(device, outputBuffer, 12*4, false)

    expect(expected).toStrictEqual(output);

});

test('transpose', async () => {
    for(let i = 1 ; i < 64 ; i++) {
        for(let j = 1 ; j < 8 ; j++) {
            let data = new Float32Array(i*j);
            for(let k = 0 ; k < i*j ; k++) {
                data[k] = k;
            }
            const expected = new Float32Array(j*i);
            for(let k = 0 ; k < i ; k++) {
                for(let l = 0 ; l < j ; l++) {
                    expected[l*i + k] = data[k*j + l];
                }
            }
            const inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
            const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(i*j));
            const transposeShader = new Ops.TransposeShader(i,j);
        
            await transposeShader.setup(device, inputBuffer, outputBuffer);
        
            createPass([transposeShader]);
            const output = await GPUUtils.writeToCPU(device, outputBuffer, i*j*4, false)
        
            expect(expected).toStrictEqual(output);
        }
    }
});

test('sum_basic', async () => {
    let data = new Float32Array([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    ])
    let expected = new Float32Array([10, 26, 42])

    const inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
    const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(3));

    const addShader = new Ops.Sum2DShader(4, 3);
    await addShader.setup(device, inputBuffer, outputBuffer);

    createPass([addShader])

    const output = await GPUUtils.writeToCPU(device, outputBuffer, 3*4, false)
    expectArraysToBeCloseTo(expected, output, 1e-6);
});

test('sum_1', async() => {
    let M = 3; 
    const PHI = 1.618033;
    
    let LEN = M;

    function sum2d(array, M:number, N: number) {
        let output = new Float32Array(N);
        for (let i = 0 ; i < N ; i++) {
            let sum = 0;
            for (let j = 0 ; j < M ; j++) {
                sum += array[i*M + j];
            }
            output[i] = sum;
        }
        return output
    }

    for(let t = 0 ; t < 16 ; t++) {
        LEN = Math.floor(LEN*PHI); 
        let data = new Float32Array(LEN*LEN);
        let sum = 0;
        for(let i = 0 ; i < LEN*LEN ; i++) {
            // random float
            data[i] = 1;
            sum += data[i];
        }
        let expected = sum2d(data, LEN, LEN)
        let inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(LEN));
        let addShader = new Ops.Sum2DShader(LEN, LEN);
        await addShader.setup(device, inputBuffer, outputBuffer);
        // console.log("nTPB ", Ops.Sum2DShader.nTPB)
        createPass([addShader]);
        await device.queue.onSubmittedWorkDone();
        const output = await GPUUtils.writeToCPU(device, outputBuffer, LEN*4, false)
        expectArraysToBeCloseTo(expected, output, 1e-6);
        inputBuffer.destroy();
        outputBuffer.destroy();
    }
});

test('sum_long', async() => {
    let M = 4; 
    const PHI = 1.618033;
    
    let LEN = M;

    function sum2d(array, M:number, N: number) {
        let output = new Float32Array(N);
        for (let i = 0 ; i < N ; i++) {
            let sum = 0;
            for (let j = 0 ; j < M ; j++) {
                sum += array[i*M + j];
            }
            output[i] = sum;
        }
        return output
    }

    const WIDTH = 33;
    for(let t = 0 ; t < 15 ; t++) {
        LEN = Math.floor(LEN*PHI); 
        let data = new Float32Array(LEN*WIDTH);
        let sum = 0;
        for(let i = 0 ; i < LEN*WIDTH ; i++) {
            data[i] = i % (LEN+1);
            sum += data[i];
        }
        let expected = sum2d(data, LEN, WIDTH)
        let inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(WIDTH));
        let addShader = new Ops.Sum2DShader(LEN, WIDTH);
        await addShader.setup(device, inputBuffer, outputBuffer);
        createPass([addShader]);
        await device.queue.onSubmittedWorkDone();
        const output = await GPUUtils.writeToCPU(device, outputBuffer, WIDTH*4, false)
        expectArraysToBeCloseTo(expected, output, 1e-6);
        inputBuffer.destroy();
        outputBuffer.destroy();
    }

})

test('add_tall', async() => {
    let M = 4; 
    const PHI = 1.618033;
    
    let LEN = M;

    function sum2d(array, M:number, N: number) {
        let output = new Float32Array(N);
        for (let i = 0 ; i < N ; i++) {
            let sum = 0;
            for (let j = 0 ; j < M ; j++) {
                sum += array[i*M + j];
            }
            output[i] = sum;
        }
        return output
    }

    const WIDTH = 2;
    for(let t = 0 ; t < 14 ; t++) {
        LEN = Math.floor(LEN*PHI); 
        let data = new Float32Array(LEN*WIDTH);
        let sum = 0;
        for(let i = 0 ; i < LEN*WIDTH ; i++) {
            data[i] = (i % (LEN+1))/10000;
            sum += data[i];
        }
        let expected = sum2d(data, LEN, WIDTH)
        let inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(WIDTH));
        let addShader = new Ops.Sum2DShader(LEN, WIDTH);
        await addShader.setup(device, inputBuffer, outputBuffer);
        createPass([addShader]);
        await device.queue.onSubmittedWorkDone();
        const output = await GPUUtils.writeToCPU(device, outputBuffer, WIDTH*4, false)
        expectArraysToBeCloseTo(expected, output, 1e-4);
        inputBuffer.destroy();
        outputBuffer.destroy();
}
   
});

test('matvecelement_add', async() => {
    let matrix = new Float32Array([1,2,3,4,5,6,7,8,9,10,11,12])
    let vector = new Float32Array([1,2,3])

    let matrixBuffer = GPUUtils.createStorageBuffer(device, matrix);
    let vectorBuffer = GPUUtils.createStorageBuffer(device, vector);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(12));
    let addShader = new Ops.MatVecElementwiseShader(3, 4);

    await addShader.setup(device, matrixBuffer, vectorBuffer, outputBuffer);
    createPass([addShader]);
    await device.queue.onSubmittedWorkDone();
    const output = await GPUUtils.writeToCPU(device, outputBuffer, 12*4, false)
    let expected = new Float32Array([2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]);
    expectArraysToBeCloseTo(expected, output, 1e-6);
});

test('matvecelement_mul', async() => {
    let matrix = new Float32Array([1,2,3,4,5,6,7,8,9,10,11,12])
    let vector = new Float32Array([1,2,3])

    let matrixBuffer = GPUUtils.createStorageBuffer(device, matrix);
    let vectorBuffer = GPUUtils.createStorageBuffer(device, vector);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(12));
    let mulShader = new Ops.MatVecElementwiseShader(3, 4, 2);

    await mulShader.setup(device, matrixBuffer, vectorBuffer, outputBuffer);
    createPass([mulShader]);
    await device.queue.onSubmittedWorkDone();
    const output = await GPUUtils.writeToCPU(device, outputBuffer, 12*4, false)
    let expected = new Float32Array([1, 2, 3, 4, 10, 12, 14, 16, 27, 30, 33, 36]);
    expect(output).toEqual(expected);
})

test('matvecelement_mul', async() => {
    let matrix = new Float32Array([1,2,3,4,5,6,7,8,9,10,11,12])
    let vector = new Float32Array([1,2,3])

    let matrixBuffer = GPUUtils.createStorageBuffer(device, matrix);
    let vectorBuffer = GPUUtils.createStorageBuffer(device, vector);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(12));
    let mulShader = new Ops.MatVecElementwiseShader(3, 4, 3);

    await mulShader.setup(device, matrixBuffer, vectorBuffer, outputBuffer);
    createPass([mulShader]);
    await device.queue.onSubmittedWorkDone();
    const output = await GPUUtils.writeToCPU(device, outputBuffer, 12*4, false)
    let expected = new Float32Array([1, 2, 3, 4, 5/2, 6/2, 7/2, 8/2, 9/3, 10/3, 11/3, 12/3]);
    expectArraysToBeCloseTo(expected, output, 1e-6);
})

test('unsorted_segment_sum_basic', async() => {
    let data = new Float32Array([1,2,3,4,5,6,7,8]);
    let segmentIds = new Uint32Array([0,0,1,1,1,1,0,2]);
    let expected = new Float32Array([10, 18, 8]);

    let dataBuffer = GPUUtils.createStorageBuffer(device, data);
    let segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
    let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(3));
    let unsortedSegmentSumShader = new Ops.UnsortedSegmentSumShader(8, 3);

    await unsortedSegmentSumShader.setup(device, dataBuffer, segmentIdsBuffer, outputBuffer);
    createPass([unsortedSegmentSumShader]);

    const output = await GPUUtils.writeToCPU(device, outputBuffer, 3*4, false);
    expect(output).toEqual(expected)
    dataBuffer.destroy();
    segmentIdsBuffer.destroy();
    outputBuffer.destroy();
});

test('unsorted_segment_sum', async() => {
    let M = 3; 
    const PHI = 1.618033;
    
    let LEN = M;

    function segmentedSum(array, segmentIds, K: number) {
        let output = new Float32Array(K);
        for (let i = 0 ; i < array.length ; i++) {
            output[segmentIds[i]] += array[i];
        }
        return output
    }

    let K = 4;
    for(let t = 0 ; t < 28 ; t++) {
        LEN = Math.floor(LEN*PHI); 
        let data = new Float32Array(LEN);
        let segmentIds = new Uint32Array(LEN);
        for(let i = 0 ; i < LEN ; i++) {
            data[i] = i % 3;
            segmentIds[i] = i % K;
        }
        let expected = segmentedSum(data, segmentIds, K)
        let inputBuffer = GPUUtils.createStorageBuffer(device, data);
        let segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds)
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
        let shader = new Ops.UnsortedSegmentSumShader(LEN, K);
        await shader.setup(device, inputBuffer, segmentIdsBuffer, outputBuffer);
        createPass([shader]);
        await device.queue.onSubmittedWorkDone();
        const output = await GPUUtils.writeToCPU(device, outputBuffer, K*4, false)
        expect(output).toEqual(expected)
        inputBuffer.destroy();
        outputBuffer.destroy();
    }
});
