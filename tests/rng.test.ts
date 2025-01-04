import {expect, test} from 'vitest'
import {beforeAll} from 'vitest'
import {Random} from '../src/utils/rng'
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

test('uniform', async () => {
    for(let t = 0 ; t < 10 ; t++) {
        const M = 100000;
        const seed = new Uint32Array([0,0,0,t])

        const seedUniformBuffer = GPUUtils.createUniform(device, seed);
        const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(M));

        const shader = new Random.UniformShader(M)
        await shader.setup(device, seedUniformBuffer, outputBuffer);

        createPass([shader]);

        const output = await GPUUtils.writeToCPU(device, outputBuffer, M*4, false)
        const bins = new Array(10).fill(0)
        for(let i = 0 ; i < output.length ; i++) {
            bins[Math.floor(output[i]*10)]++;
        }

        for(let i = 0 ; i < bins.length ; i++) {
            bins[i] = bins[i] / M;
            // assert that bins is within 0.05 of 0.1
            expect(bins[i]).toBeGreaterThan(0.1-0.05);
            expect(bins[i]).toBeLessThan(0.1+0.05);
        }

        seedUniformBuffer.destroy();
        outputBuffer.destroy();
    }
})

test('normal', async () => {
    for(let t = 0 ; t < 10 ; t++) {
        const M = 1000000;
        const seed = new Uint32Array([0,0,0,t])

        const seedUniformBuffer = GPUUtils.createUniform(device, seed);
        const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(M));

        const shader = new Random.NormalShader(M)
        await shader.setup(device, seedUniformBuffer, outputBuffer);

        createPass([shader]);

        const output = await GPUUtils.writeToCPU(device, outputBuffer, M*4, false)

        let bins = new Array(4).fill(0)
        for(let i = 0 ; i < output.length ; i++) {
            if (output[i] >= -2 && output[i] <= 2) {
                bins[Math.floor((output[i]+2))]++;
            }
        }

        for(let i = 0 ; i < bins.length ; i++) {
            bins[i] = bins[i] / M;
        }

        expect(bins[0]).toBeGreaterThan(0.136-0.01);
        expect(bins[0]).toBeLessThan(0.136+0.05);

        expect(bins[1]).toBeGreaterThan(0.341-0.01);
        expect(bins[1]).toBeLessThan(0.341+0.01);

        expect(bins[2]).toBeGreaterThan(0.341-0.01);
        expect(bins[2]).toBeLessThan(0.341+0.01);

        expect(bins[3]).toBeGreaterThan(0.136-0.01);
        expect(bins[3]).toBeLessThan(0.136+0.01);

        seedUniformBuffer.destroy();
        outputBuffer.destroy();

    }
})

// test('categorical', async () => {
// })