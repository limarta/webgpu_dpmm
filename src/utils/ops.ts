import Sum2DCode from '../shaders/algebra/sum_2d.wgsl';
import UnsortedSegmentSumCode from '../shaders/algebra/unsorted_segment_sum.wgsl';
import Sum3DCode from '../shaders/algebra/sum_3d.wgsl'
import {ShaderEncoder} from './shader.ts';
import {GPUUtils} from './gpu.ts'

namespace Ops {

export class Sum2DShader implements ShaderEncoder {
    M: number;
    N: number;
    nTPB:number;
    MAX_BLOCKS_X: number;
    BUFFER_LEN_1: number;
    BUFFER_LEN_2: number;

    columnSizes: number[];

    dimensionUniformBuffer: GPUBuffer
    workBuffer1: GPUBuffer;
    workBuffer2: GPUBuffer;
    outputBuffer: GPUBuffer;

    pipeline: GPUComputePipeline;
    bindGroup1: GPUBindGroup;
    bindGroup2: GPUBindGroup;
    shaderCode: GPUShaderModule;

    static UNIFORM_STRIDE:number = 256;
    static nTPB:number = 16;

    /**
     * 
     * @param M - number of rows of matrix
     * @param N  - number of columns of matrix
     * @returns 
     */
    constructor(
        M:number, 
        N:number, 
    ) {
        this.M = M;
        this.N = N;


        let nTPB = Sum2DShader.nTPB;
        this.MAX_BLOCKS_X = Math.ceil(N/nTPB)
        this.BUFFER_LEN_1 = M * N;
        this.BUFFER_LEN_2 = M * this.MAX_BLOCKS_X;

        this.columnSizes = [];
        for(let size=N; size > 0; size = Math.floor(size/nTPB)){
            this.columnSizes.push(size);
        }

    }

    async setup(device:GPUDevice, inputBuffer:GPUBuffer, outputBuffer: GPUBuffer) {
        let M = this.M;
        let N = this.N;

        if (inputBuffer.size / 4 != M*N) {
            throw new Error(`inputBuffer size must be equal to M*N, but got ${inputBuffer.size / 4} and ${M*N}`);
        }
        
        if (outputBuffer.size / 4 != M) {
            throw new Error(`outputBuffer size must be equal to M, but got ${outputBuffer.size / 4} and ${M}`);
        }
        this.workBuffer1 = inputBuffer;
        this.outputBuffer = outputBuffer;
        this.shaderCode = device.createShaderModule({
            code: Sum2DCode
        });

        this.dimensionUniformBuffer = device.createBuffer({
            label: "dimension_uniform",
            size: Sum2DShader.UNIFORM_STRIDE*5, // assumption: at most 5 reductions
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        for(let i = 0 ; i < this.columnSizes.length; i++) {
            device.queue.writeBuffer(this.dimensionUniformBuffer, i*Sum2DShader.UNIFORM_STRIDE, new Uint32Array([this.columnSizes[i], this.M]));
        }

        this.workBuffer2 = GPUUtils.createStorageBuffer(
            device, 
            new Float32Array(this.BUFFER_LEN_2)
        )

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                        hasDynamicOffset: true,
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
                },
            ]
        });
    
    
        this.bindGroup1 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 8
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.workBuffer1
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.workBuffer2
                    }
                },
            ]
        });
    
        this.bindGroup2 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 8
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.workBuffer2
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.workBuffer1
                    }
                },
            ]
        });
    
        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });
    
        const pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: this.shaderCode,
                entryPoint: "sum_2d_within_block",
                constants: {
                    nTPB: this.nTPB,
                }
            } 
        });

        this.pipeline = pipeline;
    
    }

    encode(pass:GPUComputePassEncoder) {
        pass.setPipeline(this.pipeline);
        for(let i = 0 ; i < this.columnSizes.length; i++) {
            if (i % 2 ==0) {
                var bg = this.bindGroup1;
            } else {
                var bg = this.bindGroup2;
            }
            pass.setBindGroup(0, bg, [i*Sum2DShader.UNIFORM_STRIDE,]);
            var workgroups = Math.ceil(this.columnSizes[i]/this.nTPB);
            pass.dispatchWorkgroups(workgroups, this.M, 1);
        }
    }
}

export class UnsortedSegmentSumShader implements ShaderEncoder {
    N: number;
    num_segments: number;
    N_intermediate: number;

    inputBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;
    segmentIdBuffer: GPUBuffer;
    segmentCountUniformBuffer: GPUBuffer;
    scratchBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    sum2DShader: Sum2DShader;

    static nTPB: number = 16;

    /**
     * 
     * @param N  - number of entries in the input array
     */
    constructor(N: number) {
        this.N = N
        this.N_intermediate = Math.ceil(N / UnsortedSegmentSumShader.nTPB);
        this.sum2DShader = new Sum2DShader(this.num_segments, this.N_intermediate);
    }

    async setup(device: GPUDevice, inputBuffer: GPUBuffer, segmentIdBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (inputBuffer.size / 4 != this.N) {
            throw new Error(`dataBuffer size must be equal to N, but got ${inputBuffer.size / 4} and ${this.N}`);
        }

        if (inputBuffer.size != segmentIdBuffer.size) {
            throw new Error(`dataBuffer size must be equal to segmentIdBuffer size, but got ${inputBuffer.size} and ${segmentIdBuffer.size}`);
        }

        if (outputBuffer.size != this.num_segments) {
            throw new Error(`outputBuffer size must be equal to num_segments, but got ${outputBuffer.size} and ${this.num_segments}`);
        }

        this.inputBuffer = inputBuffer;
        this.segmentIdBuffer = segmentIdBuffer;
        this.outputBuffer = outputBuffer;

        this.segmentCountUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.N, this.num_segments]));

        this.scratchBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.N_intermediate * this.num_segments));
        await this.sum2DShader.setup(device, this.scratchBuffer, this.outputBuffer);

        const bindGroupLayout = device.createBindGroupLayout({
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
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
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
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.inputBuffer,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.segmentIdBuffer,
                    } 
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.segmentCountUniformBuffer,
                    } 
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.scratchBuffer
                    }
                }
            ]
        });
        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });
    
        const computeShaderModule = device.createShaderModule({
            code: UnsortedSegmentSumCode,
        });
        const pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeShaderModule,
                entryPoint: "main",
            },
        });
    }

    encode(pass:GPUComputePassEncoder) {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(this.N_intermediate, this.num_segments, 1);
        this.sum2DShader.encode(pass);
    }
}

export class Sum3DShader implements ShaderEncoder {
    M: number;
    N: number;
    K: number;
    MAX_BLOCKS_X: number;
    columnSizes: number[];

    dimensionUniformBuffer: GPUBuffer;
    inputBuffer: GPUBuffer;
    scratchBuffer_1: GPUBuffer;
    scratchBuffer_2: GPUBuffer;
    outputBuffer: GPUBuffer;

    bindGroup_0: GPUBindGroup; // input -> scratch_1
    bindGroup_1: GPUBindGroup; // scratch_1 -> scratch_2
    bindGroup_2: GPUBindGroup; // scratch_2 -> scratch_1
    bindGroup_3: GPUBindGroup; // scratch_1 -> output
    bindGroup_4: GPUBindGroup; // scratch_2 -> output
    bindGroup_5: GPUBindGroup; // input -> output
    pipeline: GPUComputePipeline;

    shaderCode: GPUShaderModule

    isSetup: boolean = false;

    static nTPB: number = 4;

    constructor(M: number, N: number, K: number) {
        this.M = M;
        this.N = N;
        this.K = K;
        this.MAX_BLOCKS_X = Math.ceil(M/Sum3DShader.nTPB)

        this.columnSizes = [];
        for(let size=M; size > 0; size = Math.ceil(size/Sum3DShader.nTPB)){
            this.columnSizes.push(size);
            if (size == 1) {
                break;
            }
        }
    }

    async setup(device: GPUDevice, inputBuffer: GPUBuffer, outputBuffer: GPUBuffer, axis: number=0) {
        if (inputBuffer.size / 4 != this.M * this.N * this.K) {
            throw new Error(`inputBuffer size must be equal to M*N*K, but got ${inputBuffer.size / 4} and ${this.M * this.N * this.K}`);
        }

        if (outputBuffer.size / 4 != this.N * this.K) {
            throw new Error(`outputBuffer size must be equal to N*K, but got ${outputBuffer.size / 4} and ${this.N * this.K}`);
        }

        this.inputBuffer = inputBuffer;
        this.outputBuffer = outputBuffer;

        this.dimensionUniformBuffer = device.createBuffer({
            label: "dimension_uniform",
            size: Sum2DShader.UNIFORM_STRIDE*5, // assumption: at most 5 reductions
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });


        var width = this.M;
        for(let i = 0 ; i < this.columnSizes.length; i++) {
            device.queue.writeBuffer(this.dimensionUniformBuffer, i*Sum2DShader.UNIFORM_STRIDE, new Uint32Array([width, this.N, this.K]));
            width = Math.ceil(width / Sum3DShader.nTPB);
        }

        this.scratchBuffer_1 = GPUUtils.createStorageBuffer(device, new Float32Array(this.MAX_BLOCKS_X * this.N * this.K));
        this.scratchBuffer_2 = GPUUtils.createStorageBuffer(device, new Float32Array(this.MAX_BLOCKS_X * this.N * this.K));

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                        hasDynamicOffset: true,
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
        })
        this.bindGroup_0 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 12,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.inputBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.scratchBuffer_1,
                    }
                }
            ]
        });

        this.bindGroup_1 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 12,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.scratchBuffer_1,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.scratchBuffer_2,
                    }
                }
            ]
        });

        this.bindGroup_2 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 12,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.scratchBuffer_2,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.scratchBuffer_1,
                    }
                }
            ]
        });

        this.bindGroup_3 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 12,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.scratchBuffer_1,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.outputBuffer,
                    }
                }
            ]
        });

        this.bindGroup_4 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 12,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.scratchBuffer_2,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.outputBuffer,
                    }
                }
            ]
        });

        this.bindGroup_5 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimensionUniformBuffer,
                        size: 12,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.inputBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.outputBuffer,
                    }
                }
            ]
        });

        this.shaderCode = device.createShaderModule({
            code: Sum3DCode
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: this.shaderCode,
                entryPoint: "sum3d",
                constants: {
                    nTPB: Sum3DShader.nTPB,
                }
            }
        });

        this.isSetup = true;
    }
    
    encode(pass:GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("Sum3DShader is not setup");
        }

        pass.setPipeline(this.pipeline);
        if (this.M <= Sum3DShader.nTPB) {
            pass.setBindGroup(0, this.bindGroup_5, [0]);
            pass.dispatchWorkgroups(1,this.N, this.K);
            return;
        }

        pass.setBindGroup(0, this.bindGroup_0, [0]);
        pass.dispatchWorkgroups(this.columnSizes[1], this.N, this.K);

        var lastBuffer = true;
        console.log(this.columnSizes)
        for (let i = 1 ; i < this.columnSizes.length-1; i++) {
            let NUM_WORKGROUPS = this.columnSizes[i+1];
            console.log("Width ", NUM_WORKGROUPS)
            if (i%2 == 1) {
                console.log("I")
                lastBuffer = false;
                pass.setBindGroup(0, this.bindGroup_1, [i*Sum2DShader.UNIFORM_STRIDE]);
            } else {
                console.log("J")
                lastBuffer = true;
                pass.setBindGroup(0, this.bindGroup_2, [i*Sum2DShader.UNIFORM_STRIDE]);    
            }
            pass.dispatchWorkgroups(NUM_WORKGROUPS, this.N, this.K);
        }

        if (lastBuffer) {
            pass.setBindGroup(0, this.bindGroup_3, [(this.columnSizes.length-1)*Sum2DShader.UNIFORM_STRIDE]);
        } else {
            pass.setBindGroup(0, this.bindGroup_4, [(this.columnSizes.length-1)*Sum2DShader.UNIFORM_STRIDE]);
        }
        pass.dispatchWorkgroups(1, this.N, this.K)

    }
}

}

export {Ops};