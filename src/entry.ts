import Plotly from 'plotly.js-dist';
import {KMeansShader, kmeans} from './k_means/kmeans.ts'
import { GaussianMixtureModelShader } from './gmm/gmm.ts';
import {GPUUtils} from './utils/gpu.ts';
import {Ops} from './utils/ops.ts';
import {Random} from './utils/rng.ts';

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

  // let N = 20000;
  // let gammaShader = new Random.GammaShader(N, 9.0, 0.5);
  // let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N));
  // let seedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array([3,1,0,0]));
  // await gammaShader.setup(device, seedBuffer, outputBuffer);

  // const encoder = device.createCommandEncoder();
  // const pass = encoder.beginComputePass();
  // gammaShader.encode(pass);
  // pass.end();
  // device.queue.submit([encoder.finish()]);

  // let output = await GPUUtils.writeToCPU(device, gammaShader.outputBuffer, N * 4, false);
  // // console.log(JSON.stringify(Array.from(output), null, 2));
  // GPUUtils.log(device, gammaShader.bitBuffer, true);

  // const trace = {
  //   x: Array.from(output),
  //   type: 'histogram',
  //   histnorm: 'probability density',
  //   marker: {
  //     color: 'blue',
  //   },
  // };

  // const layout = {
  //   title: 'Probability Density Function',
  //   xaxis: { title: 'Value' },
  //   yaxis: { title: 'Density' },
  // };

  // Plotly.newPlot('test', [trace], layout);
  

  let M = 1000; // Number of samples
  let N = 2; // Number of features
  let K = 3; // Number of clusters
  
  let seedBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]));
  let proportionsBuffer = GPUUtils.createStorageBuffer(device, new Float32Array([Math.log(0.05), Math.log(.7), Math.log(0.25)]));
  let assignmentBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(M));
  let outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(M*N));
  let meansBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(
    [0.0, 10.0, -10.0, 
      0.0, 10.0, -5.0]
  ));
  let covarianceBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(
    [1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0]
  ));

  const gmmShader = new GaussianMixtureModelShader(M, N, K);
  await gmmShader.setup(
    device,
    seedBuffer,
    proportionsBuffer,
    meansBuffer,
    covarianceBuffer,
    outputBuffer,
    assignmentBuffer
  )
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  gmmShader.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);

  let assignments = await GPUUtils.writeToCPU(device, gmmShader.assignmentBuffer, M * 4);
  let data = await GPUUtils.writeToCPU(device, gmmShader.outputBuffer, M * N * 4, false);
  // // console.log(data.slice(0, M))
  // // console.log(data.slice(M, 2*M))

  // // // const data = new Float32Array(output);
  // // // const segmentIds = new Uint32Array(assignments);

  const trace = {
    x: data.slice(0, M),
    y: data.slice(M, 2*M),
    mode: 'markers',
    marker: {
      color: assignments,
      colorscale: 'Viridis',
      size: 10,
    },
    type: 'scatter',
  };

  const layout = {
    title: 'Scatter plot of data points',
    xaxis: { title: 'X' },
    yaxis: { title: 'Y' },
  };

  Plotly.newPlot('test', [trace], layout);

  // let data = new Float32Array(M*N);
  // for(let i = 0 ; i < M*N ; i++) {
  //   data[i] = Math.random();
  // }

  // let segmentIds = new Uint32Array(M);
  // for(let i = 0 ; i < M ; i++) {
  //   segmentIds[i] = Math.floor(Math.random()*K);
  // }
  // let results = await kmeans(device, M, N, K, data, segmentIds);
  // let segmentIdsInferred = results.segmentIds;
  // let means = results.means;


}

function plot_kmeans(data, segmentIds, means, M, N, K: number) {
  const centroidsTrace = {
    x: means.slice(0, K),
    y: means.slice(K, 2 * K),
    mode: 'markers',
    marker: {
      color: 'red',
      symbol: 'x',
      size: 12,
    },
    type: 'scatter',
  };
  

  const trace = {
    x: data.slice(0, M),
    y: data.slice(M, 2 * M),
    mode: 'markers',
    marker: {
      color: segmentIds,
      colorscale: 'Viridis',
      size: 10,
    },
    type: 'scatter',
  };

  const layout = {
    title: 'Scatter plot of data points',
    xaxis: { title: 'X' },
    yaxis: { title: 'Y' },
  };

  Plotly.newPlot('test', [trace, centroidsTrace], layout);

}