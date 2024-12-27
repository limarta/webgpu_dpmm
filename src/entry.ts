import Plotly from 'plotly.js-dist';
import {DPMM} from './k_means/kmeans.ts'
import {GPUUtils} from './utils/gpu.ts'
import {Ops} from './utils/ops.ts'

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


  let M = 63;
  let N = 4; 
  let K = 1;
  const sumShader = new Ops.Sum3DShader(M,N,K)
  const data = new Float32Array(M * N * K);
  for (let i = 0; i < data.length; i++) {
    data[i] = i+1;
  }


  const inputBuffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  device.queue.writeBuffer(inputBuffer, 0, data);

  const outputBuffer = device.createBuffer({
    size: 4*N*K,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  await sumShader.setup(device, inputBuffer, outputBuffer)

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  sumShader.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // console.log("scratch 1:")
  await GPUUtils.log(device, sumShader.scratchBuffer_1, false);
  // console.log("scratch 2:")
  await GPUUtils.log(device, sumShader.scratchBuffer_2, false);
  // console.log("output:")
  await GPUUtils.log(device, outputBuffer, false);
  // const K = 10;
  // const kmeans = new DPMM.KMeans(K);
  // await kmeans.setup(device);
  // kmeans.step(device);
  // GPUUtils.log(device, kmeans.numericals.muRandBuffer, true)
  // GPUUtils.log(device, kmeans.numericals.muBuffer, false)

  // const plotData = [{
  //   x: samples,
  //   type: 'histogram',
  //   xbins: {
  //     // start: 0,
  //     // end: 1,
  //     // size: (1 - 0) / 10
  //   },
  //   histnorm: 'probability'
  // }];

  // const layout = {
  //   title: 'Histogram of Mapped RNG',
  //   xaxis: { title: 'Value' },
  //   yaxis: { title: 'Probability' },
  // };

  // Plotly.newPlot('test', plotData, layout).then((gd) => {
  //   return Plotly.toImage(gd, { format: 'png', width: 800, height: 600 });
  // });
}