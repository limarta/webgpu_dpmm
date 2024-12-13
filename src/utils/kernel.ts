import {GPUUtils} from './gpu'
import SumKernelCode from '../shaders/kernels/sum.wgsl'

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
}

export {Kernels};



