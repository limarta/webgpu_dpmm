import TransposeCode from '../shaders/algebra/transpose.wgsl';
import MatVecElementwiseCode from '../shaders/algebra/mat_vec_elementwise.wgsl';
import Sum2DCode from '../shaders/algebra/sum_2d.wgsl';
import UnsortedSegmentSumCode from '../shaders/algebra/unsorted_segment_sum.wgsl';
import UnsortedSegmentSum2DCode from '../shaders/algebra/unsorted_segment_sum2d.wgsl';
import Sum3DCode from '../shaders/algebra/sum_3d.wgsl'
import {ShaderEncoder} from './shader.ts';
import {GPUUtils} from './gpu.ts'

enum OpType {
    PLUS = 0,
    MINUS = 1,
    TIMES = 2,
    DIVIDE = 3,
}

namespace Ops {

export class TransposeShader implements ShaderEncoder {
    M: number;
    N: number;

    dimsUniformBuffer: GPUBuffer
    inputBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;
    
    isSetup: boolean = false;

    static nTPB: number = 4;

    constructor(M: number, N: number) {
        this.M = M;
        this.N = N;
    }

    async setup(device: GPUDevice, inputBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (inputBuffer.size / 4 != this.M * this.N) {
            throw new Error(`inputBuffer size must be equal to M*N, but got ${inputBuffer.size / 4} and ${this.M * this.N}`);
        }
        if(inputBuffer.size != outputBuffer.size) {
            throw new Error(`inputBuffer size must be equal to outputBuffer size, but got ${inputBuffer.size} and ${outputBuffer.size}`);
        }

        this.inputBuffer = inputBuffer;
        this.outputBuffer = outputBuffer;
        this.dimsUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.M, this.N]));
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding:0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding:1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding:2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                },
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
                        buffer: this.inputBuffer
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

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: device.createShaderModule({
                    code: TransposeCode
                }),
                entryPoint: "transpose",
                constants: {
                    nTPB: TransposeShader.nTPB
                }
            }
        })

        this.isSetup = true;
    }

    encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        let MAX_BLOCKS_X = Math.ceil(this.M/TransposeShader.nTPB);
        let MAX_BLOCKS_Y = Math.ceil(this.N/TransposeShader.nTPB);
        pass.dispatchWorkgroups(MAX_BLOCKS_X, MAX_BLOCKS_Y, 1)
    }
}

/**
 * A shader to perform elementwise operations on a matrix and a vector.
 */
export class MatVecElementwiseShader implements ShaderEncoder {
    M: number;
    N: number;
    op: OpType;

    dimsUniformBuffer: GPUBuffer;
    matrixBuffer: GPUBuffer;
    vectorBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    isSetup:boolean = false;

    static nTPB: number = 4;

    constructor(M:number , N: number, op:OpType = OpType.PLUS) {
        this.M = M;
        this.N = N;
        this.op = op;
    }

    async setup(device:GPUDevice, matrixBuffer:GPUBuffer, vectorBuffer:GPUBuffer, outputBuffer:GPUBuffer) {
        if (matrixBuffer.size / 4 != this.M * this.N) {
            throw new Error(`matrixBuffer size must be equal to M*N, but got ${matrixBuffer.size / 4} and ${this.M * this.N}`);
        }
        if(vectorBuffer.size / 4 != this.M) {
            throw new Error(`vectorBuffer size must be equal to M, but got ${vectorBuffer.size / 4} and ${this.M}`);
        }
        if(outputBuffer.size /4 != this.M * this.N) {
            throw new Error(`outputBuffer size must be equal to M*N, but got ${outputBuffer.size / 4} and ${this.M * this.N}`);
        }

        this.matrixBuffer = matrixBuffer;
        this.vectorBuffer = vectorBuffer;
        this.outputBuffer = outputBuffer;

        this.dimsUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.M, this.N]));

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
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimsUniformBuffer,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.matrixBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.vectorBuffer,
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

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        const shaderCode = device.createShaderModule({
            code: MatVecElementwiseCode
        });

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderCode,
                entryPoint: "main",
                constants: {
                    nTPB: MatVecElementwiseShader.nTPB,
                    op: this.op
                }
            }
        });

        this.isSetup = true;

    }

    encode(pass:GPUComputePassEncoder): void {
        if (!this.isSetup) {
            throw new Error("MatVecElementwiseShader is not setup");
        }
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.M * this.N / MatVecElementwiseShader.nTPB), 1);
    }

}

export class Sum2DShader implements ShaderEncoder {
    M: number;
    N: number;
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

    static UNIFORM_STRIDE: number = 256;
    static nTPB: number = 32;

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
        this.MAX_BLOCKS_X = Math.ceil(M/nTPB)

        this.columnSizes = [];
        for(let size=M; size > 0; size = Math.ceil(size/nTPB)){
            this.columnSizes.push(size);
            if (size == 1) {
                break;
            }
        }

    }

    async setup(device:GPUDevice, inputBuffer:GPUBuffer, outputBuffer: GPUBuffer) {
        if (inputBuffer.size / 4 != this.M * this.N) {
            throw new Error(`inputBuffer size must be equal to M*N, but got ${inputBuffer.size / 4} and ${this.M * this.N}`);
        }

        if (outputBuffer.size / 4 != this.N) {
            throw new Error(`outputBuffer size must be equal to N, but got ${outputBuffer.size / 4} and ${this.N}`);
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
            device.queue.writeBuffer(this.dimensionUniformBuffer, i*Sum2DShader.UNIFORM_STRIDE, new Uint32Array([width, this.N]));
            width = Math.ceil(width / Sum2DShader.nTPB);
        }

        this.scratchBuffer_1 = GPUUtils.createStorageBuffer(device, new Float32Array(this.MAX_BLOCKS_X * this.N));
        this.scratchBuffer_2 = GPUUtils.createStorageBuffer(device, new Float32Array(this.MAX_BLOCKS_X * this.N));

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

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        this.shaderCode = device.createShaderModule({
            code: Sum2DCode
        });

        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: this.shaderCode,
                entryPoint: "sum2d",
                constants: {
                    nTPB: Sum2DShader.nTPB,
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
            pass.dispatchWorkgroups(1,this.N);
            return;
        }

        pass.setBindGroup(0, this.bindGroup_0, [0]);
        pass.dispatchWorkgroups(this.columnSizes[1], this.N);

        var lastBuffer = true;
        for (let i = 1 ; i < this.columnSizes.length-1; i++) {
            let NUM_WORKGROUPS = this.columnSizes[i+1];
            if (i%2 == 1) {
                lastBuffer = false;
                pass.setBindGroup(0, this.bindGroup_1, [i*Sum2DShader.UNIFORM_STRIDE]);
            } else {
                lastBuffer = true;
                pass.setBindGroup(0, this.bindGroup_2, [i*Sum2DShader.UNIFORM_STRIDE]);    
            }
            pass.dispatchWorkgroups(NUM_WORKGROUPS, this.N);
        }

        if (lastBuffer) {
            pass.setBindGroup(0, this.bindGroup_3, [(this.columnSizes.length-1)*Sum2DShader.UNIFORM_STRIDE]);
        } else {
            pass.setBindGroup(0, this.bindGroup_4, [(this.columnSizes.length-1)*Sum2DShader.UNIFORM_STRIDE]);
        }
        pass.dispatchWorkgroups(1, this.N)

        // pass.setPipeline(this.pipeline);
        // for(let i = 0 ; i < this.columnSizes.length; i++) {
        //     if (i % 2 ==0) {
        //         var bg = this.bindGroup1;
        //     } else {
        //         var bg = this.bindGroup2;
        //     }
        //     pass.setBindGroup(0, bg, [i*Sum2DShader.UNIFORM_STRIDE,]);
        //     var workgroups = Math.ceil(this.columnSizes[i]/Sum2DShader.nTPB);
        //     pass.dispatchWorkgroups(workgroups, this.M, 1);
        // }
    }
}

export class UnsortedSegmentSumShader implements ShaderEncoder {
    N: number;
    num_segments: number;
    N_intermediate: number;

    dimsUniformBuffer:GPUBuffer;
    segmentCountUniformBuffer: GPUBuffer;
    inputBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;
    segmentIdBuffer: GPUBuffer;
    scratchBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    sum2DShader: Sum2DShader;

    isSetup: boolean = false;

    static nTPB: number = 32;

    /**
     * 
     * @param N  - number of entries in the input array
     * @param num_segments - number of segments
     */
    constructor(N: number, num_segments: number) {
        this.N = N
        this.num_segments = num_segments;
        this.N_intermediate = Math.ceil(N / UnsortedSegmentSumShader.nTPB);
        this.sum2DShader = new Sum2DShader(this.N_intermediate, this.num_segments);
    }

    async setup(device: GPUDevice, inputBuffer: GPUBuffer, segmentIdBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (inputBuffer.size / 4 != this.N) {
            throw new Error(`dataBuffer size must be equal to N, but got ${inputBuffer.size / 4} and ${this.N}`);
        }

        if (inputBuffer.size != segmentIdBuffer.size) {
            throw new Error(`dataBuffer size must be equal to segmentIdBuffer size, but got ${inputBuffer.size} and ${segmentIdBuffer.size}`);
        }

        if (outputBuffer.size/4 != this.num_segments) {
            throw new Error(`outputBuffer size must be equal to num_segments, but got ${outputBuffer.size} and ${this.num_segments}`);
        }

        this.inputBuffer = inputBuffer;
        this.segmentIdBuffer = segmentIdBuffer;
        this.outputBuffer = outputBuffer;

        this.dimsUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.N]));
        this.segmentCountUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.num_segments]));
        this.scratchBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.N_intermediate * this.num_segments));


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
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimsUniformBuffer,
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
                        buffer: this.segmentIdBuffer,
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
        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeShaderModule,
                entryPoint: "main",
                constants: {
                    nTPB: UnsortedSegmentSumShader.nTPB,
                }
            },
        });

        await this.sum2DShader.setup(device, this.scratchBuffer, this.outputBuffer);
        this.isSetup = true;
    }

    encode(pass:GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("UnsortedSegmentSumShader is not setup");
        }
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

    static UNIFORM_STRIDE = 256;
    static nTPB: number = 32;

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
            size: Sum3DShader.UNIFORM_STRIDE*5, // assumption: at most 5 reductions
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });


        var width = this.M;
        for(let i = 0 ; i < this.columnSizes.length; i++) {
            device.queue.writeBuffer(this.dimensionUniformBuffer, i*Sum3DShader.UNIFORM_STRIDE, new Uint32Array([width, this.N, this.K]));
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
        for (let i = 1 ; i < this.columnSizes.length-1; i++) {
            let NUM_WORKGROUPS = this.columnSizes[i+1];
            if (i%2 == 1) {
                lastBuffer = false;
                pass.setBindGroup(0, this.bindGroup_1, [i*Sum2DShader.UNIFORM_STRIDE]);
            } else {
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

export class UnsortedSegmentSum2DShader implements ShaderEncoder {
    M: number;
    N: number;
    num_segments: number;
    M_intermediate: number;

    dimsUniformBuffer:GPUBuffer;
    segmentCountUniformBuffer: GPUBuffer;
    inputBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;
    segmentIdBuffer: GPUBuffer;
    scratchBuffer: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;

    sum3DShader: Sum3DShader;

    isSetup: boolean = false;

    static nTPB: number = 4;

    /**
     * feature contiguous (i.e. each data point has its attributes in contiguous memory)
     * 
     * @param M  - number of entries in the input array
     * @param N  - number of entries in the input array
     * @param num_segments - number of segments
     */
    constructor(M: number, N: number, num_segments: number) {
        this.M = M;
        this.N = N;
        this.num_segments = num_segments;
        this.M_intermediate = Math.ceil(M / UnsortedSegmentSum2DShader.nTPB);
        this.sum3DShader = new Sum3DShader(this.M_intermediate, this.N, this.num_segments);
    }

    async setup(device: GPUDevice, inputBuffer: GPUBuffer, segmentIdBuffer: GPUBuffer, outputBuffer: GPUBuffer) {
        if (inputBuffer.size / 4 != this.M * this.N) {
            throw new Error(`dataBuffer size must be equal to N, but got ${inputBuffer.size / 4} and ${this.M * this.N}`);
        }

        if (segmentIdBuffer.size /4 != this.M) {
            throw new Error(`segmentIdBuffer size must be equal to M, but got ${segmentIdBuffer.size} and ${this.M}`);
        }

        if (outputBuffer.size/4 != this.N * this.num_segments) {
            throw new Error(`outputBuffer size must be equal to num_segments, but got ${outputBuffer.size} and ${this.N * this.num_segments}`);
        }

        this.inputBuffer = inputBuffer;
        this.segmentIdBuffer = segmentIdBuffer;
        this.outputBuffer = outputBuffer;

        this.dimsUniformBuffer = GPUUtils.createUniform(device, new Uint32Array([this.M, this.N]));
        this.scratchBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.M_intermediate * this.N * this.num_segments));

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
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dimsUniformBuffer,
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
                        buffer: this.segmentIdBuffer,
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
            code: UnsortedSegmentSum2DCode,
        });
        this.pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeShaderModule,
                entryPoint: "main",
                constants: {
                    nTPB: UnsortedSegmentSum2DShader.nTPB,
                }
            },
        });

        await this.sum3DShader.setup(device, this.scratchBuffer, this.outputBuffer);
        this.isSetup = true;
    }

    encode(pass:GPUComputePassEncoder) {
        if (!this.isSetup) {
            throw new Error("UnsortedSegmentSumShader is not setup");
        }
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(this.M_intermediate, this.N, this.num_segments);
        this.sum3DShader.encode(pass);
    }
}

}

export {Ops};