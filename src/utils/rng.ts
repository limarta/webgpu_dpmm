import ThreeFryCode from '../shaders/rng/threefry.wgsl';
import BoxMullerCode from '../shaders/rng/boxmuller.wgsl';
import UniformCode from '../shaders/rng/uniform.wgsl';
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
        this.threeFryShader.encode(pass);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        let N_workgroups = Math.ceil(this.N/this.nTPB);
        pass.dispatchWorkgroups(N_workgroups,1,1);
    }
}

// export class NormalShader implements ShaderEncoder {
//     readonly length: number;
//     readonly bufferLength: number;
//     nTPB: number;

//     rngBuffer: GPUBuffer;
//     lengthBuffer: GPUBuffer;
//     outputBuffer: GPUBuffer;

//     bindGroup_1: GPUBindGroup;
//     pipeline_1: GPUComputePipeline;
//     bindGroupLayout_2: GPUBindGroupLayout;
//     bindGroup_2: GPUBindGroup;
//     pipeline_2: GPUComputePipeline;

//     constructor(length: number, nTPB: number=32) {
//         this.length = length
//         this.bufferLength = Math.ceil(length / 4);
//         this.nTPB = nTPB;
//     }

//     async setup(device: GPUDevice, seedBuffer:GPUBuffer, outputBuffer:GPUBuffer) {
//         let rngBuffer = this.rngBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.bufferLength));

//         this.bindGroupLayout_1 = device.createBindGroupLayout({
//             label: "ThreeFry BGL",
//             entries: [
//                 {
//                     binding: 0,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: 'uniform'
//                     }
//                 },
//                 {
//                     binding: 1,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: 'uniform'
//                     }
//                 },
//                 {
//                     binding: 2,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: 'storage'
//                     }
//                 }
//             ]
//         });

//         this.bindGroup_1 = device.createBindGroup({
//             layout: this.bindGroupLayout_1,
//             entries: [
//                 {
//                     binding: 0,
//                     resource: {
//                         buffer: seedBuffer
//                     }
//                 },
//                 {
//                     binding: 1,
//                     resource: {
//                         buffer: nBuffer
//                     }
//                 },
//                 {
//                     binding: 2,
//                     resource: {
//                         buffer: rngBuffer
//                     }
//                 }
//             ]
//         });

//         const pipelineLayout_1 = device.createPipelineLayout({
//             bindGroupLayouts: [this.bindGroupLayout_1]
//         });

//         const threeFryShader = device.createShaderModule({
//             code: ThreeFryCode
//         })
//         this.pipeline_1 = device.createComputePipeline({
//             layout: pipelineLayout_1,
//             compute: {
//                 module: threeFryShader,
//                 entryPoint: "threefry",
//                 constants: {
//                     nTPB: 16,
//                 }
//             } 
//         })

//         this.bindGroupLayout_2 = device.createBindGroupLayout({
//             label: "BoxMuller BGL",
//             entries: [
//                 {
//                     binding: 0,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: 'uniform'
//                     }
//                 },
//                 {
//                     binding: 1,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: 'read-only-storage'
//                     }
//                 },
//                 {
//                     binding: 2,
//                     visibility: GPUShaderStage.COMPUTE,
//                     buffer: {
//                         type: 'storage'
//                     }
//                 }
//             ]
//         });

//         this.bindGroup_2 = device.createBindGroup({
//             layout: this.bindGroupLayout_2,
//             entries: [
//                 {
//                     binding: 0,
//                     resource: {
//                         buffer: nBuffer
//                     }
//                 },
//                 {
//                     binding: 1,
//                     resource: {
//                         buffer: rngBuffer
//                     }
//                 },
//                 {
//                     binding: 2,
//                     resource: {
//                         buffer: outputBuffer
//                     }
//                 }
//             ]
//         });

//         const pipelineLayout_2 = device.createPipelineLayout({
//             bindGroupLayouts: [this.bindGroupLayout_2]
//         });

//         const boxMullerShader = device.createShaderModule({
//             code: BoxMullerCode
//         })
//         this.pipeline_2 = device.createComputePipeline({
//             layout: pipelineLayout_2,
//             compute: {
//                 module: boxMullerShader,
//                 entryPoint: "boxmuller",
//                 constants: {
//                     nTPB: 16,
//                 }
//             } 
//         });

//     }

//     // BUG: If 2^n +2, then last two entries are 0???
//     encode(pass:GPUComputePassEncoder) {
//         pass.setPipeline(this.pipeline_1);
//         pass.setBindGroup(0, this.bindGroup_1);
//         pass.dispatchWorkgroups(Math.ceil(this.length/this.nTPB),1,1);

//         pass.setPipeline(this.pipeline_2);
//         pass.setBindGroup(0, this.bindGroup_2);
//         pass.dispatchWorkgroups(Math.ceil(this.length/this.nTPB),1,1);
//     }
// }


}

export {Random};