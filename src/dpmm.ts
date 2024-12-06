import { GPUUtils } from './utils/gpu.ts';
import randomWGSL from './shaders/random.wgsl'
import Plotly from 'plotly.js-dist';


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

  const N = 256;
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
  passEncoder.dispatchWorkgroups(256, 1, 1)
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

  const plotData = [{
    x: data,
    type: 'histogram',
    xbins: {
      size: 0.1,
    }
  }];

  const layout = {
    title: 'Histogram of Data',
    xaxis: { title: 'Value' },
    yaxis: { title: 'Count' },
  };

  const plotHtml = Plotly.newPlot('test', plotData, layout).then((gd) => {
    return Plotly.toImage(gd, { format: 'png', width: 800, height: 600 });
  });
  // const plotHtml = Plotly.newPlot('plot', plotData, layout).then((gd) => {
  //   return Plotly.toImage(gd, { format: 'png', width: 800, height: 600 });
  // });
}
