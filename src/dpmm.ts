import { GPUUtils } from './utils/gpu.ts';
import randomWGSL from './shaders/distributions/random.wgsl'
import Plotly from 'plotly.js-dist';
import {Kernels} from './utils/kernel.ts'
import {Random} from './utils/rng.ts'

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

  const seed = new Uint32Array([0,0,0,1]);
  const N = 1000000;
  const rng = await Random.threefry(device, seed, N);
  const mappedRng = new Float32Array(N*4);
  for (let i = 0; i < N*4; i++) {
    mappedRng[i] = rng[i] / 0xFFFFFFFF;
  }
  const plotData = [{
    x: mappedRng,
    type: 'histogram',
    xbins: {
      start: 0,
      end: 1,
      size: (1 - 0) / 10
    },
    histnorm: 'probability'
  }];

  const layout = {
    title: 'Histogram of Mapped RNG',
    xaxis: { title: 'Value' },
    yaxis: { title: 'Probability' },
  };

  Plotly.newPlot('test', plotData, layout).then((gd) => {
    return Plotly.toImage(gd, { format: 'png', width: 800, height: 600 });
  });
  // const N = 4096;
  // const K = 10;
  // const seedInit = Array.from({length: 4*N}, ()=>Math.floor(Math.random()*100000));

  // const M = 3;
  // const N = 32*32*32;
  // const arr = new Float32Array(Array.from({length: N*M}, (_, i) => 1));
  // const segment_ids = new Uint32Array(Array.from({ length: N*M }, (_, i) => i % 2))
  // const segment_ids = new Uint32Array(Array.from({ length: 96 }, (_, i) => 0))
  // await Kernels.unsorted_segment_sum(device, arr, segment_ids, 2)
  // await Kernels.sum_2d(device, arr, M, N);


}

type DPMM = {
  seed: GPUBuffer,
  rand: GPUBuffer,
  pi: GPUBuffer,
  mu: GPUBuffer,
  c: GPUBuffer,
  data: GPUBuffer,
  n: GPUBuffer,
  k: GPUBuffer,
  _n: number
  _k: number
}

function dpmmInit(device: GPUDevice, N: number, K: number, seedInit:number[] | null): DPMM {
  const seedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(seedInit));
  const randBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N));
  const piBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
  const muBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
  const cBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N));
  const dataBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N));
  const nUniform = GPUUtils.createUniform(device, new Uint32Array(1));
  const kUniform = GPUUtils.createUniform(device, new Uint32Array(1));
  return {
    seed: seedBuffer,
    rand: randBuffer,
    pi: piBuffer,
    mu: muBuffer,
    c: cBuffer,
    data: dataBuffer, 
    n: nUniform,
    k: kUniform,
    _n: N,
    _k: 0
  };
}

async function addData(device: GPUDevice, dpmm: DPMM, data: Float32Array) {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      { // input array
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
      { // scratch space?
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform"
        }
      }, {
        binding: 3, // store final answer?
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform"
        }
      }
    ]
  });

  const scratch = GPUUtils.createStorageBuffer(device, new Float32Array(data.length));
  const bind_group = GPUUtils.createBindGroup(device, bindGroupLayout, [dpmm.data, scratch, dpmm.n, dpmm.k]);
}

async function sampleNormal(device: GPUDevice, dpmm: DPMM) {
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

  const bind_group = GPUUtils.createBindGroup(device, bindGroupLayout, [dpmm.seed, dpmm.rand]);
  const randomShader = GPUUtils.createShaderModule(device, randomWGSL, "random", [16,1,1])

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
    size: dpmm.rand.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(
    dpmm.rand, 0,
    srcBuffer, 0,
    dpmm.rand.size
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
    },
    histnorm: 'probability'
  }];

  const layout = {
    title: 'Histogram of Data',
    xaxis: { title: 'Value' },
    yaxis: { title: 'Count' },
  };

  const plotHtml = Plotly.newPlot('test', plotData, layout).then((gd) => {
    return Plotly.toImage(gd, { format: 'png', width: 800, height: 600 });
  });

}