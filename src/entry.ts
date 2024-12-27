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


  let N = 16;
  let K = 2;
  let data = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    data[i] = i+1;
  }
  let segmentIds = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    segmentIds[i] = i % K;
  }

  const dataBuffer = GPUUtils.createStorageBuffer(device, data);
  const segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
  const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));

  const segmentedShader = new Ops.UnsortedSegmentSumShader(N, K);

  await segmentedShader.setup(device, dataBuffer, segmentIdsBuffer, outputBuffer)

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  segmentedShader.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // console.log("scratch 1:")
  await GPUUtils.log(device, segmentedShader.scratchBuffer, false);
  // console.log("scratch 2:")
  // await GPUUtils.log(device, sumShader.scratchBuffer_2, false);
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