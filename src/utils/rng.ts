import ThreeFryCode from '../shaders/rng/threefry.wgsl';
import BoxMullerCode from '../shaders/rng/boxmuller.wgsl';

namespace Random {

interface ShaderEncoder {
    encode(pass:GPUComputePassEncoder):void;
}

export class NormalShaderEncoder implements ShaderEncoder {
    length: number;
    nTPB: number = 16;
    bindGroupLayout_1: GPUBindGroupLayout;
    bindGroup_1: GPUBindGroup;
    pipeline_1: GPUComputePipeline;
    bindGroupLayout_2: GPUBindGroupLayout;
    bindGroup_2: GPUBindGroup;
    pipeline_2: GPUComputePipeline;

    constructor(device:GPUDevice, length: number, seedBuffer:GPUBuffer, nBuffer:GPUBuffer, rngBuffer:GPUBuffer, outputBuffer:GPUBuffer) {
        if ((rngBuffer.size/4) % 4 != 0) {
            throw new Error(`rngBuffer size must be a multiple of 4, but got ${rngBuffer.size / 4}`);
        }

        if (length != rngBuffer.size/16) {
            throw new Error(`length must be equal to rngBuffer size / 4, but got ${length} and ${rngBuffer.size / 4}`);
        }

        this.length = length;

        this.bindGroupLayout_1 = device.createBindGroupLayout({
            label: "ThreeFry BGL",
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
                        type: 'uniform'
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

        this.bindGroup_1 = device.createBindGroup({
            layout: this.bindGroupLayout_1,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: seedBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: nBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: rngBuffer
                    }
                }
            ]
        });

        const pipelineLayout_1 = device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout_1]
        });

        const threeFryShader = device.createShaderModule({
            code: ThreeFryCode
        })
        this.pipeline_1 = device.createComputePipeline({
            layout: pipelineLayout_1,
            compute: {
                module: threeFryShader,
                entryPoint: "threefry",
                constants: {
                    nTPB: 16,
                }
            } 
        })

        this.bindGroupLayout_2 = device.createBindGroupLayout({
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

        this.bindGroup_2 = device.createBindGroup({
            layout: this.bindGroupLayout_2,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: nBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: rngBuffer
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

        const pipelineLayout_2 = device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout_2]
        });

        const boxMullerShader = device.createShaderModule({
            code: BoxMullerCode
        })
        this.pipeline_2 = device.createComputePipeline({
            layout: pipelineLayout_2,
            compute: {
                module: boxMullerShader,
                entryPoint: "boxmuller",
                constants: {
                    nTPB: 16,
                }
            } 
        });

    }

    // BUG: If 2^n +2, then last two entries are 0???
    encode(pass:GPUComputePassEncoder) {
        pass.setPipeline(this.pipeline_1);
        pass.setBindGroup(0, this.bindGroup_1);
        pass.dispatchWorkgroups(Math.ceil(this.length/this.nTPB),1,1);

        pass.setPipeline(this.pipeline_2);
        pass.setBindGroup(0, this.bindGroup_2);
        pass.dispatchWorkgroups(Math.ceil(this.length/this.nTPB),1,1);
    }
}


}

export {Random};