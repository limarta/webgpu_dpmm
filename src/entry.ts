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

  let M_1 = 1000;
  let M_2 = 4;
  let K = 2;
  let data_1 = new Float32Array(M_1*K);
  for (let i = 0; i < data_1.length; i++) {
    // data_1[i] = Math.random();
    data_1[i] = 0;
  }
  let data_2 = new Float32Array(M_2*K);
  for (let i = 0; i < data_2.length; i++) {
    data_2[i] = 2;
  }
  function closetsPoints(arr_1, arr_2) {
    const indices = new Uint32Array(arr_1.length);
    for (let i = 0 ; i < arr_1.length ; i++) {
        var bestDist = Number.MAX_VALUE;
        var bestIndex = -1;
        for(let j = 0 ;j < arr_2.length ; j++) {
          if (Math.abs(arr_1[i] - arr_2[j]) < bestDist) {
            bestDist = Math.abs(arr_1[i] - arr_2[j]);
            bestIndex = j;
          }
        }
        indices[i] = bestIndex;
    }
    console.log(indices);
  }
  closetsPoints(data_1, data_2);

  const dataBuffer_1 = GPUUtils.createStorageBuffer(device, data_1);
  const dataBuffer_2 = GPUUtils.createStorageBuffer(device, data_2);
  const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(M_1));

  const shader = new Ops.ClosestPairwiseLoopShader(M_1, M_2, K);
  await shader.setup(device, dataBuffer_1, dataBuffer_2, outputBuffer);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  shader.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  GPUUtils.log(device, outputBuffer, false);

  // let M = 56;
  // let N = 4;
  // let K = 11;
  // // let data = new Float32Array([
  // //   100, 100, 100, 100,
  // //   2, 2, 2, 2,
  // //   -1, -1, -1, -1,
  // // ]);
  // let data = new Float32Array(M*N);
  // for(let i = 0 ; i < M*N ; i++) {
  //   data[i] = i % 3;
  // }

  // let segmentIds = new Uint32Array(M);
  // for (let i = 0; i < M; i++) {
  //   segmentIds[i] = i % K;
  // }
  // let counts = new Float32Array(K);
  // for (let i = 0 ; i < K; i++) {
  //   counts[i] = 10.0;
  // }

  // const dataBuffer = GPUUtils.createStorageBuffer(device, data);
  // const segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
  // const countsBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(K));
  // const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));
  // const outputBuffer2 = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));

  // const shader1 = new Ops.UnsortedSegmentSum2DShader(M, N, K);
  // const shader2 = new Ops.TransposeShader(N, K);
  // const shader3 = new Ops.CountShader(M, K)
  // const shader4 = new Ops.MatVecElementwiseShader(K, N, 3)

  // await shader1.setup(device, dataBuffer, segmentIdsBuffer, outputBuffer);
  // await shader2.setup(device, outputBuffer, outputBuffer2)
  // await shader3.setup(device, segmentIdsBuffer, countsBuffer)
  // await shader4.setup(device, outputBuffer2, countsBuffer, outputBuffer)


  // const encoder = device.createCommandEncoder();
  // const pass = encoder.beginComputePass();
  // shader1.encode(pass);
  // shader2.encode(pass);
  // shader3.encode(pass);
  // shader4.encode(pass);
  // pass.end();
  // device.queue.submit([encoder.finish()]);

  // GPUUtils.log(device, dataBuffer, false);
  // GPUUtils.log(device, segmentIdsBuffer, true);
  // GPUUtils.log(device, countsBuffer, false)
  // GPUUtils.log(device, outputBuffer2, false);
  // GPUUtils.log(device, outputBuffer, false);



  // let M = 5;
  // let N = 4;
  // let K = 2;
  // let data = new Float32Array(M*N);
  // for (let i = 0; i < M*N; i++) {
  //   data[i] = i+1;
  // }
  // let segmentIds = new Uint32Array(M);
  // for (let i = 0; i < M; i++) {
  //   segmentIds[i] = i % K;
  // }

  // const dataBuffer = GPUUtils.createStorageBuffer(device, data);
  // const segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);
  // const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*K));

  // const segmentedShader = new Ops.UnsortedSegmentSum2DShader(M, N, K);

  // await segmentedShader.setup(device, dataBuffer, segmentIdsBuffer, outputBuffer)

  // const encoder = device.createCommandEncoder();
  // const pass = encoder.beginComputePass();
  // segmentedShader.encode(pass);
  // pass.end();
  // device.queue.submit([encoder.finish()]);

  // console.log("scratch 1:")
  // await GPUUtils.log(device, segmentedShader.scratchBuffer, false);
  // console.log("scratch 2:")
  // await GPUUtils.log(device, sumShader.scratchBuffer_2, false);
  // console.log("output:")
  // await GPUUtils.log(device, outputBuffer, false);
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