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


}

export {Random};