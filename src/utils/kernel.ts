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
        }
    ]
  });

  const arrayBuffer = GPUUtils.createStorageBuffer(device, arr, true)
  const _ans = new Float32Array(256);
  const ansBuffer = GPUUtils.createStorageBuffer(device, _ans);
  const copyBuffer = device.createBuffer({
    size: _ans.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  const bindGroup = device.createBindGroup({
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
                buffer: ansBuffer
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

  const encoder = device.createCommandEncoder();

  const pass = encoder.beginComputePass();
  pass.setBindGroup(0, bindGroup);
  pass.setPipeline(pipeline);
  let start = performance.now();
  pass.dispatchWorkgroups(256, 1, 1)
  pass.end()


  encoder.copyBufferToBuffer(
    ansBuffer, 0,
    copyBuffer, 0, _ans.byteLength
  )

  const gpuCommands = encoder.finish()
  device.queue.submit([gpuCommands])
  let duration = performance.now() - start;

  await copyBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(copyBuffer.getMappedRange());
  console.log(data);
  console.log(`Duration ${duration.toFixed(2)} ms`);

  start = performance.now()
  const s = arr.reduce((s, cur) => s + cur, 0)
  console.log("Sum s ", s)
  duration = performance.now() - start;
  console.log(`Duration ${duration.toFixed(2)} ms`);

}
}

export {Kernels};



