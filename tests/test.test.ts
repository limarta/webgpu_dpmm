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
    expect(expected).toEqual(output)
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

    for(let t = 0 ; t < 6 ; t++) {
        LEN = Math.floor(LEN*PHI); 
        let data = new Float32Array(LEN*LEN);
        for(let i = 0 ; i < LEN*LEN ; i++) {
            data[i] = i % (LEN+1);
        }
        let expected = sum2d(data, LEN, LEN)
        let inputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(data));
        let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(LEN));
        let addShader = new Ops.Sum2DShader(LEN, LEN);
        await addShader.setup(device, inputBuffer, outputBuffer);
        createPass([addShader]);
        const output = await GPUUtils.writeToCPU(device, outputBuffer, LEN*4, false)
        // custom message
        expect(expected, `LEN ${LEN}`).toEqual(output)
        inputBuffer.destroy();
        outputBuffer.destroy();
    }
})
