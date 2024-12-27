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


  let M = 4;
  let N = 4;
  let K = 4;
  const sumShader = new Ops.Sum3DShader(M,N,K)
  const data = new Float32Array([
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    4, 4, 4, 4,
    4, 4, 4, 4,
    4, 4, 4, 4,
    4, 4, 4, 4,
  ]);

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

  GPUUtils.log(device, inputBuffer, false);
  GPUUtils.log(device, outputBuffer, false);
  GPUUtils.log(device, sumShader.scratchBuffer_1, false);
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