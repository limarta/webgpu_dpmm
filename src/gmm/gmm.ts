import {Random} from '../utils/rng.ts';
import {GPUUtils} from '../utils/gpu.ts';
import {Ops} from '../utils/ops.ts';
import {ShaderEncoder} from '../utils/shader.ts'


class GaussianMixtureModelShader implements ShaderEncoder {
    M: number;
    N: number;
    K: number;
    nTPB: number;

    dimensionsBuffer: GPUBuffer;
    seedBuffer: GPUBuffer;
    subSeedBuffer: GPUBuffer;
    categoricalSeedBuffer: GPUBuffer;
    normalSeedBuffer: GPUBuffer;
    proportionBuffer: GPUBuffer;
    meanBuffer: GPUBuffer;
    covarianceBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;
    assignmentBuffer: GPUBuffer;

    threeFryShader: Random.ThreeFryShader;
    copyShader1: Random.CopyKeyShader;
    copyShader2: Random.CopyKeyShader;
    normalShader: Random.NormalShader;
    scaleAndShiftShader: Ops.ScaleAndShiftIndexed2DShader;
    categoricalShader: Random.CategoricalShader;

    isSetup: boolean = false;

    /**
     * 
     * @param M - Number of samples
     * @param N - Number of dimensions
     * @param K  - Number of components
     * @param nTPB - Number of threads per block
     */
    constructor(M: number, N: number, K: number, nTPB: number = 32) {
        this.M = M;
        this.N = N;
        this.K = K;
        this.nTPB = nTPB;

        this.threeFryShader = new Random.ThreeFryShader(4+4);
        this.copyShader1 = new Random.CopyKeyShader(0);
        this.copyShader2 = new Random.CopyKeyShader(1);
        this.categoricalShader = new Random.CategoricalShader(this.M, this.K);
        this.scaleAndShiftShader = new Ops.ScaleAndShiftIndexed2DShader(this.M, this.N, this.K);
        this.normalShader = new Random.NormalShader(this.K * this.M);
    }

    async setup(
        device: GPUDevice, 
        seedBuffer: GPUBuffer, 
        proportionBuffer: GPUBuffer,
        meanBuffer: GPUBuffer, 
        covarianceBuffer: GPUBuffer, 
        outputBuffer: GPUBuffer, 
        assignmentBuffer: GPUBuffer, 
    ) {

        if (proportionBuffer.size / 4 != this.K) {
            throw new Error(`proportionBuffer size must be equal to ${this.K}, but got ${proportionBuffer.size / 4}`);
        }

        if (meanBuffer.size / 4 != this.N * this.K) {
            throw new Error(`meanBuffer size must be equal to ${this.K * this.N}, but got ${meanBuffer.size / 4}`);
        }

        if (covarianceBuffer.size / 4 != this.N * this.K) {
            throw new Error(`covarianceBuffer size must be equal to ${this.K * this.N}, but got ${covarianceBuffer.size / 4}`);
        }

        if (outputBuffer.size / 4 != this.M * this.N) {
            throw new Error(`outputBuffer size must be equal to ${this.M * this.N}, but got ${outputBuffer.size / 4}`);
        }

        if (assignmentBuffer.size / 4 != this.M) {
            throw new Error(`assignmentBuffer size must be equal to ${this.M}, but got ${assignmentBuffer.size / 4}`);
        }

        this.seedBuffer = seedBuffer;
        this.subSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(8));
        this.categoricalSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4));
        this.normalSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4));
        this.proportionBuffer = proportionBuffer;
        this.meanBuffer = meanBuffer;
        this.covarianceBuffer = covarianceBuffer;
        this.outputBuffer = outputBuffer;
        this.assignmentBuffer = assignmentBuffer;

        await this.threeFryShader.setup(device, seedBuffer, this.subSeedBuffer);
        await this.copyShader1.setup(device, this.subSeedBuffer, this.categoricalSeedBuffer)
        await this.copyShader2.setup(device, this.subSeedBuffer, this.normalSeedBuffer)
        await this.categoricalShader.setup(device, this.categoricalSeedBuffer, proportionBuffer, assignmentBuffer);
        await this.normalShader.setup(device, this.normalSeedBuffer, outputBuffer);
        await this.scaleAndShiftShader.setup(
            device,
            this.outputBuffer,
            this.assignmentBuffer,
            this.covarianceBuffer,
            this.meanBuffer
        );

        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder): void {
        if (!this.isSetup) {
            throw new Error("Gaussian Mixture Model shader not setup");
        }
        this.threeFryShader.encode(pass);
        this.copyShader1.encode(pass);
        this.copyShader2.encode(pass);
        this.categoricalShader.encode(pass);
        this.normalShader.encode(pass);
        this.scaleAndShiftShader.encode(pass);
    }
}

export {GaussianMixtureModelShader};