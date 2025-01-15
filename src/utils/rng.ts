import CopyKeyCode from '../shaders/rng/copy_key.wgsl';
import ThreeFryCode from '../shaders/rng/threefry.wgsl';
import BoxMullerCode from '../shaders/rng/boxmuller.wgsl';
import UniformCode from '../shaders/rng/uniform.wgsl';
import CategoricalCode from '../shaders/rng/categorical.wgsl';
import GammaCode from '../shaders/rng/gamma.wgsl';
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
    
    destroy() {
        this.lengthBuffer.destroy();
    }
}

export class CopyKeyShader implements ShaderEncoder {
    N: number;
    index: number;

    indexUniformBuffer: GPUBuffer;
    seedBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    isSetup: boolean = false;

    constructor(index: number) {
        this.index = index;
    }

    async setup(device: GPUDevice, seedBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (seedBuffer.size / 4 < (this.index+1) * 4) {
            throw new Error(`seedBuffer size must be greater than ${(this.index+1)*4}, but got ${seedBuffer.size / 4}`);
        }
        this.indexUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.index]));
        this.seedBuffer = seedBuffer;
        this.outputBuffer = outputBuffer;

        let bindGroupLayout = device.createBindGroupLayout({
            label: "CopyKey BGL",
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
            label: "CopyKey BG",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.indexUniformBuffer,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: seedBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: outputBuffer
                    }
                }
            ]
        });

        let pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        let shaderModule = device.createShaderModule({
            code: CopyKeyCode
        });

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "main",
            }
        });

        this.isSetup = true;
    }

    encode(pass:GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("Shader not setup");
        }

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(1,1,1);
    }

    destroy() {
        this.indexUniformBuffer.destroy();
    }
}

export class UniformShader implements ShaderEncoder {
    readonly N: number
    readonly rngBufferLength: number
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
        this.rngBufferLength = Math.ceil(N / 4)*4;
        this.nTPB = nTPB;

        this.threeFryShader = new ThreeFryShader(this.rngBufferLength);
    }

    async setup(device: GPUDevice, seedUniformBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (outputBuffer.size / 4 !== this.N) {
            throw new Error(`outputBuffer size must be equal to ${this.N}, but got ${outputBuffer.size / 4}`);
        }

        this.seedUniformBuffer = seedUniformBuffer;
        this.lengthBuffer = GPUUtils.createUniform(device, new Uint32Array([this.N]));
        this.rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(this.rngBufferLength));
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

    destroy() {
        this.lengthBuffer.destroy();
        this.rngBuffer.destroy();
        this.threeFryShader.destroy();
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

    destroy() {
        this.lengthUniformBuffer.destroy();
        this.rngBuffer.destroy();
        this.threeFryShader.destroy();
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
    uniformSamplesBuffer: GPUBuffer
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
            throw new Error(`logprobsBuffer size must be equal to ${this.K}, but got ${logprobsBuffer.size /4 }`);
        }

        if (outputBuffer.size /4 !== this.N ) {
            throw new Error(`outputBuffer size must be equal to ${this.N}, but got ${outputBuffer.size /4 }`);
        }

        this.seedUniformBuffer = seedUniformBuffer;
        this.lengthUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.N, this.K]));
        this.logprobsBuffer = logprobsBuffer;
        this.uniformSamplesBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.N * this.K));
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
                        buffer: this.uniformSamplesBuffer
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


        await this.uniformShader.setup(device, seedUniformBuffer, this.uniformSamplesBuffer)

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

    destroy() {
        this.lengthUniformBuffer.destroy();
        this.uniformSamplesBuffer.destroy();
        this.uniformShader.destroy();
    }

}

export class GammaShader implements ShaderEncoder {
    N: number;
    shape: number;
    scale: number;
    nTPB: number;

    dimsUniformBuffer: GPUBuffer;
    parametersUniformBuffer: GPUBuffer;
    seedBuffer: GPUBuffer;
    subSeedBuffer: GPUBuffer;
    uniformSeedBuffer: GPUBuffer;
    normalSeedBuffer: GPUBuffer;
    uniformRngBuffer: GPUBuffer;
    normalRngBuffer: GPUBuffer;
    bitBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    threeFryShader: ThreeFryShader;
    copyShader1: CopyKeyShader;
    copyShader2: CopyKeyShader;
    normalShader: NormalShader;
    uniformShader: UniformShader;

    isSetup: boolean = false;

    constructor(N: number, shape: number, scale: number, nTPB: number = 32) {
        this.N = N;
        this.shape = shape;
        this.scale = scale;
        this.nTPB = nTPB;


        this.threeFryShader = new ThreeFryShader(8);
        this.copyShader1 = new Random.CopyKeyShader(0);
        this.copyShader2 = new Random.CopyKeyShader(1);
        this.normalShader = new NormalShader(4*this.N);
        this.uniformShader = new UniformShader(4*this.N);
    }

    async setup(device: GPUDevice, seedBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (outputBuffer.size / 4 != this.N) {
            throw new Error(`outputBuffer size must be equal to ${this.N}, but got ${outputBuffer.size / 4}`);
        }

        this.dimsUniformBuffer = GPUUtils.createUniform(device, new Float32Array([this.N]));
        this.parametersUniformBuffer = GPUUtils.createUniform(device, new Float32Array([this.shape, this.scale]));
        this.uniformRngBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(4*this.N));
        this.normalRngBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(4*this.N));
        this.bitBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(this.N))
        this.outputBuffer = outputBuffer;

        this.seedBuffer = seedBuffer;
        this.subSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(8));
        this.uniformSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4));
        this.normalSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4));

        let bindGroupLayout = device.createBindGroupLayout({
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
                        type: "uniform"
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
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        });

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimsUniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.parametersUniformBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.uniformRngBuffer,
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.normalRngBuffer,
                    }
                },
                {
                    binding: 4,
                    resource: {
                        buffer: this.bitBuffer
                    }
                },
                {
                    binding: 5,
                    resource: {
                        buffer: this.outputBuffer
                    }
                }
            ]
        });

        let pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        let shader = device.createShaderModule({
            code: GammaCode
        });

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "main",
                constants: {
                    nTPB: this.nTPB
                }
            }
        })


        await this.threeFryShader.setup(device, seedBuffer, this.subSeedBuffer);
        await this.copyShader1.setup(device, this.subSeedBuffer, this.uniformSeedBuffer);
        await this.copyShader2.setup(device, this.subSeedBuffer, this.normalSeedBuffer);
        await this.uniformShader.setup(device, this.uniformSeedBuffer, this.uniformRngBuffer);
        await this.normalShader.setup(device, this.normalSeedBuffer, this.normalRngBuffer);

        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder): void {
        this.threeFryShader.encode(pass);
        this.copyShader1.encode(pass);
        this.copyShader2.encode(pass);
        this.uniformShader.encode(pass);
        this.normalShader.encode(pass);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.N / this.nTPB), 1, 1);
    }

    destroy() {
        this.threeFryShader.destroy();
        this.copyShader1.destroy();
        this.copyShader2.destroy();
        this.uniformShader.destroy();
        this.normalShader.destroy();
    }
}

export class Gamma2DShader implements ShaderEncoder {
    M: number
    N: number
    nTPB: number

    dimsUniformBuffer: GPUBuffer;
    seedBuffer: GPUBuffer;
    subSeedBuffer: GPUBuffer;
    uniformSeedBuffer: GPUBuffer;
    normalSeedBuffer: GPUBuffer;
    uniformRNGBuffer: GPUBuffer;
    normalRNGBuffer: GPUBuffer;
    shapeBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    threeFryShader: ThreeFryShader;
    copyShader1: CopyKeyShader;
    copyShader2: CopyKeyShader;
    normalShader: NormalShader;
    uniformShader: UniformShader;

    isSetup: boolean = false;

    constructor(M: number, N: number, nTPB: number = 32) {
        this.M = M;
        this.N = N;
        this.nTPB = nTPB;

        this.threeFryShader = new ThreeFryShader(8);
        this.copyShader1 = new CopyKeyShader(0);
        this.copyShader2 = new CopyKeyShader(1);

        this.normalShader = new NormalShader(4*this.M*this.N);
        this.uniformShader = new UniformShader(4*this.M*this.N);
    }

    async setup(device: GPUDevice, seedBuffer: GPUBuffer, shapeBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        this.seedBuffer = seedBuffer;
        this.shapeBuffer = shapeBuffer;
        this.outputBuffer = outputBuffer;

        this.dimsUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.M, this.N]));
        this.subSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(8));
        this.uniformSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4));
        this.normalSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4));
        this.uniformRNGBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.M * this.N))
        this.normalRNGBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.M * this.N))

        let bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                }
            ]
        })

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        "buffer": this.dimsUniformBuffer
                    }
                }
            ]
        });

        await this.threeFryShader.setup(device, seedBuffer, this.subSeedBuffer);
        await this.copyShader1.setup(device, this.subSeedBuffer, this.uniformSeedBuffer);
        await this.copyShader2.setup(device, this.subSeedBuffer, this.normalSeedBuffer);
        await this.normalShader.setup(device, this.normalSeedBuffer, this.normalRNGBuffer)
        await this.uniformShader.setup(device, this.uniformSeedBuffer, this.uniformRNGBuffer)

        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("Shader not setup");
        }
        this.threeFryShader.encode(pass);
        this.copyShader1.encode(pass);
        this.copyShader2.encode(pass);
        this.uniformShader.encode(pass);
        this.normalShader.encode(pass);
    }

    destroy() {
        this.threeFryShader.destroy();
        this.copyShader1.destroy();
        this.copyShader2.destroy();
        this.uniformShader.destroy();
        this.normalShader.destroy();

        this.subSeedBuffer.destroy();
        this.uniformSeedBuffer.destroy();
        this.normalSeedBuffer.destroy();
        this.uniformRNGBuffer.destroy();
        this.uniformRNGBuffer.destroy();
    }
}

// export class DirichletSampler implements ShaderEncoder {
//     K: number;
//     nTPB: number;

//     dimsUniformBuffer: GPUBuffer;
//     seedBuffer: GPUBuffer;
//     subSeedBuffer: GPUBuffer;
//     outputBuffer: GPUBuffer;

//     bindGroup: GPUBindGroup
//     pipeline: GPUComputePipeline

//     constructor(K: number, nTPB: number = 32) {
//         this.K = K;
//         this.nTPB = nTPB;
//     }

//     async setup(device:GPUDevice, seedBuffer:GPUBuffer, outputBuffer:GPUBuffer) {
//         this.seedBuffer = seedBuffer;
//         this.outputBuffer = outputBuffer;

//         this.dimsUniformBuffer = GPUUtils.createUniform(device, new Float32Array([this.K]));
//         // this.normalUniformSeed = GPUUtils.createUniform(device, )
//         this.subSeedBuffer =  GPUUtils.createStorageBuffer(device, new Uint32Array(8))
//         this.uniformSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4))
//         this.uniformSeedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(4))

//         let bindGroupLayout = device.createBindGroupLayout({
//             entries: [
//                 {
//                     binding: 0,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: "uniform"
//                     }
//                 },
//                 {
//                     binding: 1,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: "read-only-storage"
//                     }
//                 },
//                 {
//                     binding: 2,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: "storage"
//                     }
//                 }
//             ]
//         })

//         this.bindGroup = device.createBindGroup({
//             layout: bindGroupLayout,
//             entries: [
//                 {
//                     binding: 0,
//                     resource: {
//                         "buffer": this.dimsUniformBuffer
//                     }
//                 }
//             ]

//         });
//     }

//     encode(pass:GPUComputePassEncoder) {

//     }

//     destroy() {
//         this.dimsUniformBuffer.destroy()
//     }


// }


}

export {Random};