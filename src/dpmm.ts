import { GPUUtils } from './utils/gpu.ts';

// Note: You can import your separate WGSL shader files like this.
// import triangleVertWGSL from './shaders/triangle.vert.wgsl';
// import fragWGSL from './shaders/red.frag.wgsl';
import randomWGSL from './shaders/random.wgsl'


export default async function init(
  context: GPUCanvasContext,
  device: GPUDevice
) {
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'opaque',
  });

  const N = 16;
  const seedInit = Array.from({length: 4*N}, ()=>Math.floor(Math.random()*100000));
  const seedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(seedInit));
  const randBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N));

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
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

  const bind_group = GPUUtils.createBindGroup(device, bindGroupLayout, [seedBuffer, randBuffer]);
  const randomShader = GPUUtils.createShaderModule(device, randomWGSL, "random", [4,1,1])

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    compute: {
      module: randomShader,
      entryPoint: "main"
    }
  });

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bind_group);
  passEncoder.dispatchWorkgroups(16, 1, 1)
  passEncoder.end();

  const srcBuffer = device.createBuffer({
    size: randBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(
    randBuffer, 0,
    srcBuffer, 0,
    randBuffer.size
  )


  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);

  await srcBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(srcBuffer.getMappedRange());
  console.log("Data: ", data);
}
