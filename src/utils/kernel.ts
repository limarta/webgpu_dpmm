import {GPUUtils} from './gpu'
import SumKernelCode from '../shaders/algebra/sum.wgsl'
import UnsortedSegmentSumCode from '../shaders/algebra/unsorted_segment_sum.wgsl'
import Sum2DKernelCode from '../shaders/algebra/sum_2d.wgsl'
import ThreeFryCode from '../shaders/rng/threefry.wgsl';
import UniformCode from '../shaders/rng/uniform.wgsl';
import BoxMullerCode from '../shaders/rng/boxmuller.wgsl';

namespace Kernels {
export class Shader {
}

export async function sum(device: GPUDevice, arr: Float32Array) {
  const computeShaderModule = device.createShaderModule({
    code: SumKernelCode,
  });
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
                type: "storage"
            }
        },
    ],
  });


  const arrayBuffer = GPUUtils.createStorageBuffer(device, arr, true)
  const N_intermediate = Math.ceil(arr.length / 128);
  const intermediateBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N_intermediate));
  const intermediateBuffer2 = GPUUtils.createStorageBuffer(device, new Float32Array(128));
  const sumBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(1), true);
  const copyBuffer = device.createBuffer({
    size: sumBuffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  const bindGroup1 = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: arrayBuffer
            }
        },
        {
            binding: 1,
            resource: {
                buffer: intermediateBuffer
            }
        }
    ]
  });

  const bindGroup2 = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: intermediateBuffer
            }
        },
        {
            binding: 1,
            resource: {
                buffer: intermediateBuffer2
            }
        }
    ]
  });

  const bindGroup3 = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: intermediateBuffer2
            }
        },
        {
            binding: 1,
            resource: {
                buffer: sumBuffer
            }
        }
    ]
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });
  

  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
        entryPoint: "sum_1d",
        module: computeShaderModule,
    }
  });

  let start = performance.now();
  for(let i = 0 ; i < 100 ; i++) {
    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup1);
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(N_intermediate, 1, 1)
    
    pass.setBindGroup(0, bindGroup2);
    pass.dispatchWorkgroups(128, 1, 1)

    pass.setBindGroup(0, bindGroup3);
    pass.dispatchWorkgroups(1, 1, 1)
    pass.end()

    const gpuCommands = encoder.finish()
    device.queue.submit([gpuCommands])
  }

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    sumBuffer, 0,
    copyBuffer, 0, 4,
  )
  const gpuCommands = encoder.finish()
  device.queue.submit([gpuCommands])

  await copyBuffer.mapAsync(GPUMapMode.READ);
  let duration = performance.now() - start;
  const data = new Float32Array(copyBuffer.getMappedRange());
  console.log(`Duration ${duration.toFixed(2)} ms`);

  start = performance.now()
  let s = 0;
  for(let i = 0 ; i < 100 ; i++) {
    s = arr.reduce((s, cur) => s + cur, 0)
  }
  console.log("Sum s ", s)
  duration = performance.now() - start;
  console.log(`Duration ${duration.toFixed(2)} ms`);

}

/**
 * Reduces each row of a matrix to its sum.
 * @param device 
 * @param arr 
 * @param M 
 * @param N 
 * @param axis 
 */
export async function sum_2d(device: GPUDevice, arr: Float32Array, M:number, N: number) {
    const nTPB = 16;
    const MAX_BLOCKS_X = Math.ceil(N/(nTPB))
    const INTERMEDIATE_LENGTH = M * MAX_BLOCKS_X;
    const COL_SIZES = [];
    for(let N_=N; N_ > 0; N_ = Math.floor(N_/nTPB)){
        COL_SIZES.push(N_);
    }

    const shaderCode = device.createShaderModule({
        code: Sum2DKernelCode
    })

    const dimensionUniform = device.createBuffer({
        label: "dimension_uniform",
        size: 256*5, // assumption: at most 5 reductions
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const UNIFORM_STRIDE = 256;
    for(let i = 0 ; i < COL_SIZES.length; i++) {
        device.queue.writeBuffer(dimensionUniform, i*UNIFORM_STRIDE, new Uint32Array([COL_SIZES[i], M]));
    }
    const inputBuffer = GPUUtils.createStorageBuffer(device, arr);
    const intermediateBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(INTERMEDIATE_LENGTH));

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


    const bindGroup1 = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: dimensionUniform,
                    size: 8
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: inputBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: intermediateBuffer
                }
            },
        ]
    });

    const bindGroup2 = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: dimensionUniform,
                    size: 8
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: intermediateBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: inputBuffer
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
            module: shaderCode,
            entryPoint: "sum_2d_within_block",
            constants: {
                nTPB: nTPB,
            }
        } 
    });

    const pipeline_final = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderCode,
            entryPoint: "sum_2d_final",
            constants: {
                nTPB: nTPB,
            }
        },
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass();
    // apply first kernel
    pass.setPipeline(pipeline);
    for(let i = 0 ; i < COL_SIZES.length; i++) {
        if (i % 2 ==0) {
            var bg = bindGroup1;
        } else {
            var bg = bindGroup2;
        }
        pass.setBindGroup(0, bg, [i*UNIFORM_STRIDE,]);
        var workgroups = Math.ceil(COL_SIZES[i]/nTPB);
        pass.dispatchWorkgroups(workgroups, M, 1);
    }

    pass.end();

    const cpuBuffer = device.createBuffer({
        size: M*4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    if (COL_SIZES.length % 2 == 0) {
        var outBuffer = inputBuffer;
    } else {
        var outBuffer = intermediateBuffer;
    }
    encoder.copyBufferToBuffer(
        outBuffer, 0,
        cpuBuffer, 0,
        M*4,
    );

    const commands = encoder.finish();

    device.queue.submit([commands])

    await cpuBuffer.mapAsync(GPUMapMode.READ)
    const ans = new Float32Array(cpuBuffer.getMappedRange());

    // for (let i = 0; i < M; i++) {
    //     console.log(ans.slice(i * MAX_BLOCKS_X, (i + 1) * MAX_BLOCKS_X));
    // }

    console.log(ans);
    return ans;

}


export async function unsorted_segment_sum(
    device: GPUDevice, 
    data:Float32Array, 
    segment_ids: Uint32Array, 
    num_segments: number
) {
    if (data.length != segment_ids.length) {
        throw RangeError(`data length ${data.length} different from segment ids ${segment_ids.length}`);
    }
    const inputBuffer = GPUUtils.createStorageBuffer(device, data);
    const segmentBuffer = GPUUtils.createStorageBuffer(device, segment_ids);
    const segmentCountUniform = GPUUtils.createUniform(device, new Uint32Array([num_segments]));

    const N_intermediate = Math.ceil(data.length / 32);
    const intermediateBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N_intermediate * num_segments));
    const sumBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(num_segments));

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

    const bindGroup1 = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: inputBuffer,
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: segmentBuffer,
                } 
            },
            {
                binding: 2,
                resource: {
                    buffer: segmentCountUniform,
                } 
            },
            {
                binding: 3,
                resource: {
                    buffer: intermediateBuffer
                }
            }
        ]
    });

    // const bindGroup2 = device.createBindGroup({
    //     layout: bindGroupLayout,
    //     entries: [
    //         {
    //             binding: 0,
    //             resource: {
    //                 buffer: intermediateBuffer,
    //             }
    //         },
    //         {
    //             binding: 1,
    //             resource: {
    //                 buffer: sumBuffer,
    //             }
    //         }
    //     ],
    // });

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


    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setBindGroup(0, bindGroup1);
    pass.setPipeline(pipeline)
    pass.dispatchWorkgroups(N_intermediate, num_segments);
    
    pass.end();

    const cpuBuffer = device.createBuffer({
        size: 4*num_segments*N_intermediate,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    encoder.copyBufferToBuffer(
        intermediateBuffer, 0,
        cpuBuffer, 0,
        4*num_segments*N_intermediate
    );

    const GPUCommands = encoder.finish();
    device.queue.submit([GPUCommands])

    await cpuBuffer.mapAsync(GPUMapMode.READ)
    const ans = new Float32Array(cpuBuffer.getMappedRange());
    console.log(ans);
    console.log("num segments ", 2)
    console.log("N_intermediate ", N_intermediate)
    sum_2d(device, ans, num_segments, N_intermediate);
}


/*
RANDOM KERNELS
*/

export async function prng(device: GPUDevice, seed: Uint32Array, N: number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*4*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        rngBuffer, 0,
        cpuBuffer, 0,
        N*4*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Uint32Array(cpuBuffer.getMappedRange())
    const result = new Uint32Array(ans);
    cpuBuffer.unmap();
    return result;
}

export async function threefry(device: GPUDevice, seedBuffer:GPUBuffer, nBuffer:GPUBuffer, rngBuffer:GPUBuffer, N: number) {
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

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
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

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        code: ThreeFryCode
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "threefry",
        }
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass()

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    let N_workgroups = Math.ceil(N/16);
    pass.dispatchWorkgroups(N_workgroups,1,1);

    pass.end();
    const commands = encoder.finish();

    device.queue.submit([commands]);
}

async function boxmuller(device:GPUDevice, rngBuffer:GPUBuffer, nBuffer:GPUBuffer, samplesBuffer:GPUBuffer, N:number) {
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

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
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
                    buffer: samplesBuffer 
                }
            }
        ]
    });
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        code: BoxMullerCode
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "boxmuller",
        }
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass()

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    let N_workgroups = Math.ceil(N/16);
    pass.dispatchWorkgroups(N_workgroups,1,1);

    pass.end();
    const commands = encoder.finish();

    device.queue.submit([commands]);
}

export async function normal(device: GPUDevice, seed: Uint32Array, N: number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*2));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    
    boxmuller(device, rngBuffer, nBuffer, outputBuffer, N)

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*2*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        outputBuffer, 0,
        cpuBuffer, 0,
        N*2*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Float32Array(cpuBuffer.getMappedRange())
    const result = new Float32Array(ans);
    console.log(result)
    cpuBuffer.unmap();
    return result;
}

async function uniform_(device:GPUDevice, rngBuffer:GPUBuffer, nBuffer:GPUBuffer, samplesBuffer:GPUBuffer, N:number) {
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

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
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
                    buffer: samplesBuffer 
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

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "uni",
        }
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass()

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    let N_workgroups = Math.ceil(N/16);
    pass.dispatchWorkgroups(N_workgroups,1,1);

    pass.end();
    const commands = encoder.finish();

    device.queue.submit([commands]);
}

export async function uniform(device:GPUDevice, seed: Uint32Array, N:number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    const samplesBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*4));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    
    uniform_(device, rngBuffer, nBuffer, samplesBuffer, N);

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*4*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        samplesBuffer, 0,
        cpuBuffer, 0,
        N*4*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Float32Array(cpuBuffer.getMappedRange())
    const result = new Float32Array(ans);
    cpuBuffer.unmap();
    cpuBuffer.destroy();
    return result;
}

export async function gumbelmax(device:GPUDevice, 
    seedBuffer:GPUBuffer, 
    nBuffer:GPUBuffer,
    rngBuffer:GPUBuffer) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();

}

export async function categorical(device:GPUDevice, seed:Uint32Array, N:number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    const samplesBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*4));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    
    uniform_(device, rngBuffer, nBuffer, samplesBuffer, N);

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*4*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        samplesBuffer, 0,
        cpuBuffer, 0,
        N*4*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Float32Array(cpuBuffer.getMappedRange())
    const result = new Float32Array(ans);
    cpuBuffer.unmap();
    cpuBuffer.destroy();
    return result;
}
}

export {Kernels};



