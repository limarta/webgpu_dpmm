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


  let M = 10;
  let N = 2;
  let K = 2;
  // let data = new Float32Array([
  //   100, 100, 100, 100,
  //   2, 2, 2, 2,
  //   -1, -1, -1, -1,
  // ]);
  // let data = new Float32Array(M*N);
  // for(let i = 0 ; i < M*N ; i++) {
  //   data[i] = i % 3;
  // }
  let data = new Float32Array(
    [0.44373084494754944, 0.012341715444441181, 0.07892681580529581, 0.16983717354013406, 0.559106625669447, 0.09920468528804882, 0.9425709791902743, 0.9632256863827882, 0.48785917694153547, 0.727832145515513, 
      0.7857840841423287, 0.4504541380106568, 0.3017574987032916, 0.2975755734713885, 0.6796020014334652, 0.37140687603818456, 0.7541709343165697, 0.7574181903211246, 0.6553652292341736, 0.7375753349071723])

  // console.log("Data:");
  // for (let i = 0; i < M; i++) {
  //   let row = "";
  //   for (let j = 0; j < N; j++) {
  //     row += data[i * N + j] + " ";
  //   }
  //   console.log(row);
  // }

  let segmentIds = new Uint32Array(M);
  for (let i = 0; i < M; i++) {
    segmentIds[i] = i % K;
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
  const shader5 = new Ops.ClosestPairwiseLoopShader(M, K, N)

  await shader1.setup(device, dataBuffer, segmentIdsBuffer, meansBuffer);
  await shader2.setup(device, meansBuffer, meansBufferTranspose);
  await shader3.setup(device, segmentIdsBuffer, countsBuffer);
  await shader4.setup(device, meansBufferTranspose, countsBuffer, meansBuffer);
  await shader5.setup(device, dataBuffer, meansBuffer, segmentIdsBuffer);


  for(let i = 0 ; i < 2 ; i++) {
  await GPUUtils.log(device, segmentIdsBuffer, true);
  await GPUUtils.log(device, meansBufferTranspose, false);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  shader1.encode(pass);
  shader2.encode(pass);
  shader3.encode(pass);
  // shader4.encode(pass);
  shader5.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  console.log("Iteration: " + i);
  }

  // GPUUtils.log(device, dataBuffer, false);
  GPUUtils.log(device, segmentIdsBuffer, true);
  // GPUUtils.log(device, countsBuffer, false)
  // GPUUtils.log(device, outputBuffer2, false);
  // GPUUtils.log(device, meansBuffer, false);
  GPUUtils.log(device, meansBufferTranspose, false);

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