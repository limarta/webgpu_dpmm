import {GPUUtils} from './gpu'
import SumKernelCode from '../shaders/kernels/sum.wgsl'
import UnsortedSegmentSumCode from '../shaders/kernels/unsorted_segment_sum.wgsl'
import Sum2DKernelCode from '../shaders/kernels/sum_2d.wgsl'

namespace Kernels {
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

    // const dimensionUniform = GPUUtils.createUniform(device, new Uint32Array([N, M]));
    const dimensionUniform = device.createBuffer({
        label: "dimension_uniform",
        size: 256*5, // assumption: at most 5 reductions
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const UNIFORM_STRIDE = 256;
    for(let i = 0 ; i < COL_SIZES.length; i++) {
        console.log(COL_SIZES[i]);
        device.queue.writeBuffer(dimensionUniform, i*UNIFORM_STRIDE, new Uint32Array([COL_SIZES[i], M]));
    }
    const inputBuffer = GPUUtils.createStorageBuffer(device, arr);


    const intermediateBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(INTERMEDIATE_LENGTH));
    const sumBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(M));

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
            {
                binding: 3,
                resource: {
                    buffer: sumBuffer
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
            {
                binding: 3,
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
    pass.setBindGroup(0, bindGroup1, [0,]);
    pass.dispatchWorkgroups(MAX_BLOCKS_X, M, 1);

    // apply final kernel
    pass.setBindGroup(0, bindGroup2, [256,])
    pass.setPipeline(pipeline_final);
    pass.dispatchWorkgroups(1, M, 1);
    pass.end();

    const cpuBuffer = device.createBuffer({
        size: M*4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    encoder.copyBufferToBuffer(
        // intermediateBuffer, 0,
        sumBuffer, 0,
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
    
    // pass.setBindGroup(0, bindGroup2);
    // pass.setPipeline(pipeline) // switch to something else?
    // pass.dispatchWorkgroups(64, 1, 1);
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
    console.log(ans)
}
}

export {Kernels};



