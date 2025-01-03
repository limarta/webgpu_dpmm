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


  let M = 100;
  let N = 2;
  let K = 16;
  
  
  let data = new Float32Array(M*N);
  for(let i = 0 ; i < M*N ; i++) {
    data[i] = Math.random();
  }

  let segmentIds = new Uint32Array(M);
  for(let i = 0 ; i < M ; i++) {
    segmentIds[i] = Math.floor(Math.random()*K);
  }

  const dataBuffer = GPUUtils.createStorageBuffer(device, data);
  const segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
  const countsBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
  const meansBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));
  const meansBufferTranspose = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));

  const shader1 = new Ops.UnsortedSegmentSum2DShader(M, N, K);
  const shader2 = new Ops.TransposeShader(N, K);
  const shader3 = new Ops.CountShader(M, K)
  const shader4 = new Ops.MatVecElementwiseShader(K, N, 3)
  const shader5 = new Ops.TransposeShader(K, N);
  const shader6 = new Ops.ClosestPairwiseLoopShader(M, K, N)

  await shader1.setup(device, dataBuffer, segmentIdsBuffer, meansBuffer);
  await shader2.setup(device, meansBuffer, meansBufferTranspose);
  await shader3.setup(device, segmentIdsBuffer, countsBuffer);
  await shader4.setup(device, meansBufferTranspose, countsBuffer, meansBuffer);
  await shader5.setup(device, meansBuffer, meansBufferTranspose);
  await shader6.setup(device, dataBuffer, meansBufferTranspose, segmentIdsBuffer);


  for(let i = 0 ; i < 200 ; i++) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  shader1.encode(pass);
  shader2.encode(pass);
  shader3.encode(pass);
  shader4.encode(pass);
  shader5.encode(pass);
  shader6.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  // console.log("Iteration: " + i);
  // await GPUUtils.log(device, segmentIdsBuffer, true);
  // await GPUUtils.log(device, meansBuffer, false);
  }
  console.log("Done")

  // GPUUtils.log(device, dataBuffer, false);
  // GPUUtils.log(device, segmentIdsBuffer, true);
  // GPUUtils.log(device, countsBuffer, false)
  // GPUUtils.log(device, outputBuffer2, false);
  // GPUUtils.log(device, meansBuffer, false);
  // GPUUtils.log(device, meansBufferTranspose, false);
  // const means = await GPUUtils.writeToCPU(device, meansBufferTranspose, K*4*N, false);

  // const dataArray = Array.from(data);
  // const segmentIdsArray = Array.from(await GPUUtils.writeToCPU(device, segmentIdsBuffer, M*4, true));
  // const meansArray = Array.from(new Float32Array(means.buffer));

  // const centroidsTrace = {
  //   x: meansArray.slice(0, K),
  //   y: meansArray.slice(K, 2 * K),
  //   mode: 'markers',
  //   marker: {
  //     color: 'red',
  //     symbol: 'x',
  //     size: 12,
  //   },
  //   type: 'scatter',
  // };
  

  // const trace = {
  //   x: dataArray.slice(0, M),
  //   y: dataArray.slice(M, 2 * M),
  //   mode: 'markers',
  //   marker: {
  //     color: segmentIdsArray,
  //     colorscale: 'Viridis',
  //     size: 10,
  //   },
  //   type: 'scatter',
  // };

  // const layout = {
  //   title: 'Scatter plot of data points',
  //   xaxis: { title: 'X' },
  //   yaxis: { title: 'Y' },
  // };

  // Plotly.newPlot('test', [trace, centroidsTrace], layout);

}