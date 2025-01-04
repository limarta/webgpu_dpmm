import Plotly from 'plotly.js-dist';
import {KMeansShader, kmeans} from './k_means/kmeans.ts'
import {GPUUtils} from './utils/gpu.ts'
import {Ops} from './utils/ops.ts'
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

  // let M = 1000;
  // let N = 2;
  // let K = 8;
  
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