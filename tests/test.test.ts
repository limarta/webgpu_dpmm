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

test('add_basic', async () => {
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

test('add_1', async() => {
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
        // await GPUUtils.log(device, addShader.scratchBuffer_1, false);
        // let sb1 = await GPUUtils.writeToCPU(device, addShader.scratchBuffer_1, LEN*4, false)
        // let sb2 = await GPUUtils.writeToCPU(device, addShader.scratchBuffer_2, LEN*4, false)
        // console.log("sum of sb1 ", sb1.reduce((a, b) => a+b, 0))
        // await GPUUtils.log(device, addShader.scratchBuffer_2, false);
        // await GPUUtils.log(device, outputBuffer, false);
        // console.log("SUM ", sum)
        expectArraysToBeCloseTo(expected, output, 1e-6);
        inputBuffer.destroy();
        outputBuffer.destroy();
    }
});

test('add_long', async() => {
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
    for(let t = 0 ; t < 17 ; t++) {
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
    for(let t = 0 ; t < 16 ; t++) {
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
