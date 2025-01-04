import ThreeFryCode from '../shaders/rng/threefry.wgsl';
import BoxMullerCode from '../shaders/rng/boxmuller.wgsl';
import UniformCode from '../shaders/rng/uniform.wgsl';
import CategoricalCode from '../shaders/rng/categorical.wgsl';
import {GPUUtils} from './gpu.ts';
import {ShaderEncoder} from './shader.ts';

namespace Random {

export class ThreeFryShader implements ShaderEncoder {
    N: number
    nTPB: number

    seedUniformBuffer: GPUBuffer
    lengthBuffer: GPUBuffer
    outputBuffer: GPUBuffer

    pipeline:GPUComputePipeline
    bindGroup:GPUBindGroup
    constructor(N: number, nTPB:number = 32) {
        this.N = N;
        this.nTPB = nTPB;
    }

    async setup(device:GPUDevice, seedUniformBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        this.seedUniformBuffer = seedUniformBuffer;
        this.lengthBuffer = GPUUtils.createUniform(device, new Float32Array([this.N]));
        this.outputBuffer = outputBuffer;

        const bindGroupLayout = device.createBindGroupLayout({
            label: "ThreeFry BGL",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        });
    
        this.bindGroup = device.createBindGroup({
            label: "ThreeFry BG",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.seedUniformBuffer 
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.lengthBuffer 
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.outputBuffer 
                    }
                }
            ]
        });
    
        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });
    
        const shaderModule = device.createShaderModule({
            code: ThreeFryCode
        });
    
        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "threefry",
            }
        });
    
    }
    encode(pass:GPUComputePassEncoder) {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        let N_workgroups = Math.ceil(this.N / this.nTPB);
        pass.dispatchWorkgroups(N_workgroups,1,1);
    }
}

export class UniformShader implements ShaderEncoder {
    readonly N: number
    readonly bufferLength: number
    nTPB: number

    seedUniformBuffer: GPUBuffer
    lengthBuffer: GPUBuffer
    rngBuffer: GPUBuffer
    outputBuffer: GPUBuffer

    pipeline: GPUComputePipeline
    bindGroup: GPUBindGroup
    threeFryShader: ThreeFryShader

    isSetup: boolean = false;

    constructor(N: number, nTPB: number = 32) {
        this.N = N;
        this.bufferLength = Math.ceil(N / 4)*4;
        this.nTPB = nTPB;

        this.threeFryShader = new ThreeFryShader(this.bufferLength);
    }

    async setup(device: GPUDevice, seedUniformBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        this.seedUniformBuffer = seedUniformBuffer;
        this.lengthBuffer = GPUUtils.createUniform(device, new Uint32Array([this.N]));
        this.rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(this.bufferLength));
        this.outputBuffer = outputBuffer

        const bindGroupLayout = device.createBindGroupLayout({
            label: "Uniform BGL",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        });
    
        this.bindGroup = device.createBindGroup({
            label: "Uniform BG",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer:  this.lengthBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.rngBuffer 
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.outputBuffer 
                    }
                }
            ]
        });
        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });
    
        const shaderModule = device.createShaderModule({
            code: UniformCode
        });
    
        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "uni",
            }
        });

        await this.threeFryShader.setup(device, seedUniformBuffer, this.rngBuffer);
        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("Shader not setup");
        }

        this.threeFryShader.encode(pass);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        let N_workgroups = Math.ceil(this.N/this.nTPB);
        pass.dispatchWorkgroups(N_workgroups,1,1);
    }
}

export class NormalShader implements ShaderEncoder {
    readonly N: number;
    readonly bufferLength: number;
    nTPB: number;

    seedUniformBuffer: GPUBuffer;
    lengthUniformBuffer: GPUBuffer;
    rngBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;

    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    threeFryShader: ThreeFryShader;

    isSetup: boolean = false;

    constructor(N: number, nTPB: number=32) {
        this.N = N;
        this.bufferLength = Math.ceil(N / 4)*4;
        this.nTPB = nTPB;

        this.threeFryShader = new ThreeFryShader(this.bufferLength);
    }

    async setup(device: GPUDevice, seedUniformBuffer:GPUBuffer, outputBuffer:GPUBuffer) {

        this.seedUniformBuffer = seedUniformBuffer;
        this.lengthUniformBuffer = GPUUtils.createUniform(device, new Float32Array([this.N]));
        this.rngBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.bufferLength));
        this.outputBuffer = outputBuffer;


        const bindGroupLayout = device.createBindGroupLayout({
            label: "BoxMuller BGL",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform'
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'read-only-storage'
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage'
                    }
                }
            ]
        });

        this.bindGroup = device.createBindGroup({
            label: "BoxMuller BG",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.lengthUniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.rngBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.outputBuffer
                    }
                }
            ]
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        const boxMullerShader = device.createShaderModule({
            code: BoxMullerCode
        })

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: boxMullerShader,
                entryPoint: "boxmuller",
                constants: {
                    nTPB: this.nTPB,
                }
            } 
        });

        await this.threeFryShader.setup(device, seedUniformBuffer, this.rngBuffer);
        this.isSetup = true;
    }

    encode(pass:GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("Shader not setup");
        }

        this.threeFryShader.encode(pass);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.N / this.nTPB),1,1);
    }
}

export class CategoricalShader implements ShaderEncoder {
    readonly N: number
    readonly K: number
    isSerial: boolean
    nTPB: number

    seedUniformBuffer: GPUBuffer
    lengthUniformBuffer: GPUBuffer
    logprobsBuffer: GPUBuffer
    rngBuffer: GPUBuffer
    outputBuffer: GPUBuffer

    pipeline: GPUComputePipeline
    bindGroup: GPUBindGroup
    uniformShader: UniformShader

    isSetup: boolean = false;

    constructor(N: number, num_labels: number, isSerial: boolean = true, nTPB: number = 32) {
        this.N = N;
        this.K = num_labels
        this.isSerial = isSerial
        this.nTPB = nTPB;
        this.uniformShader = new UniformShader(this.N*this.K);
    }

    async setup(device: GPUDevice, seedUniformBuffer: GPUBuffer, logprobsBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (logprobsBuffer.size /4 !== this.K ) {
            throw new Error(`logprobsBuffer size must be euqal to ${this.K}, but got ${logprobsBuffer.size /4 }`);
        }

        if (outputBuffer.size /4 !== this.N ) {
            throw new Error(`outputBuffer size must be euqal to ${this.N}, but got ${outputBuffer.size /4 }`);
        }

        this.seedUniformBuffer = seedUniformBuffer;
        this.lengthUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.N, this.K]));
        this.logprobsBuffer = logprobsBuffer;
        this.rngBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.N * this.K));
        this.outputBuffer = outputBuffer;

        let bindGroupGroupLayout = device.createBindGroupLayout({
            label: "Categorical BGL",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        });

        this.bindGroup = device.createBindGroup({
            label: "Categorical BG",
            layout: bindGroupGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.lengthUniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.rngBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.logprobsBuffer
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.outputBuffer
                    }
                }
            ]
        });

        let pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupGroupLayout]
        });
        let shaderModule = device.createShaderModule({
            code: CategoricalCode
        });

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "main",
                constants: {
                    nTPB: this.nTPB,
                }
            }
        });


        await this.uniformShader.setup(device, seedUniformBuffer, this.rngBuffer)

        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("Shader not setup");
        }
        this.uniformShader.encode(pass);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.N / this.nTPB),1,1)
    }

}

export class GammaShader implements ShaderEncoder {
    N: number;
    shape: number;
    scale: number;
    nTPB: number;

    normalShader: NormalShader;
    uniformShader: UniformShader;

    constructor(N: number, shape: number, scale: number, nTPB: number = 32) {
        this.N = N;
        this.shape = shape;
        this.scale = scale;
        this.nTPB = nTPB;

        this.normalShader = new NormalShader(this.N);
        this.uniformShader = new UniformShader(this.N);
    }

    async setup(device: GPUDevice) {
        throw new Error("Method not implemented.");
    }

    encode(pass: GPUComputePassEncoder): void {
        throw new Error("Method not implemented.");
    }
}

export class GaussianMixtureModelShader implements ShaderEncoder {
    M: number;
    N: number;
    K: number;
    nTPB: number;

    dimensionsBuffer: GPUBuffer;
    seedUniformBuffer: GPUBuffer;
    proportionBuffer: GPUBuffer;
    meanBuffer: GPUBuffer;
    covarianceBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;
    assignmentBuffer: GPUBuffer;

    normalShader: NormalShader;
    categoricalShader: CategoricalShader;

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

        this.categoricalShader = new CategoricalShader(this.M, this.K);
        // this.normalShader = new NormalShader(this.N * this.D);
    }

    async setup(
        device: GPUDevice, 
        seedUniformBuffer: GPUBuffer, 
        proportionBuffer: GPUBuffer,
        meanBuffer: GPUBuffer, 
        covarianceBuffer: GPUBuffer, 
        outputBuffer: GPUBuffer, 
        assignmentBuffer: GPUBuffer, 
    ) {

        this.seedUniformBuffer = seedUniformBuffer;
        this.proportionBuffer = proportionBuffer;
        this.meanBuffer = meanBuffer;
        this.covarianceBuffer = covarianceBuffer;
        this.outputBuffer = outputBuffer;
        this.assignmentBuffer = assignmentBuffer;

        await this.categoricalShader.setup(device, seedUniformBuffer, proportionBuffer, assignmentBuffer);
        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder): void {
        if (!this.isSetup) {
            throw new Error("Gaussian Mixture Model shader not setup");
        }
        this.categoricalShader.encode(pass);
    }
}


}

export {Random};