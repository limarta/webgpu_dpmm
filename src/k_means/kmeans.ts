import {Random} from '../utils/rng.ts';
import {GPUUtils} from '../utils/gpu.ts';
import {Ops} from '../utils/ops.ts';
import {ShaderEncoder} from '../utils/shader.ts'


class KMeansShader implements ShaderEncoder {
  M: number;
  N: number;
  K: number;

  dataBuffer: GPUBuffer;
  segmentIdsBuffer: GPUBuffer;
  countsBuffer: GPUBuffer;
  meansBuffer: GPUBuffer;
  meansBufferTranspose: GPUBuffer;

  shaders: Array<ShaderEncoder>;

  isSetup: boolean = false;
  /**
   * 
   * @param M - number of data points
   * @param N - number of features
   * @param K - number of clusters
   */
  constructor(M: number, N:number, K: number) {
    this.M = M;
    this.N = N;
    this.K = K;
    this.shaders = [
      new Ops.UnsortedSegmentSum2DShader(M, N, K),
      new Ops.TransposeShader(N, K),
      new Ops.CountShader(M, K),
      new Ops.MatVecElementwiseShader(K, N, 3),
      new Ops.TransposeShader(K, N),
      new Ops.ClosestPairwiseLoopShader(M, K, N),
    ];
  }

  async setup(
    device:GPUDevice, 
    dataBuffer: GPUBuffer, 
    segmentIdsBuffer: GPUBuffer, 
  ) {
    this.dataBuffer = dataBuffer;
    this.segmentIdsBuffer = segmentIdsBuffer;
    let countsBuffer = this.countsBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.K));
    let meansBuffer = this.meansBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.N*this.K));
    let meansBufferTranspose = this.meansBufferTranspose = GPUUtils.createStorageBuffer(device, new Float32Array(this.N*this.K));

    await this.shaders[0].setup(device, dataBuffer, segmentIdsBuffer, meansBuffer);
    await this.shaders[1].setup(device, meansBuffer, meansBufferTranspose);
    await this.shaders[2].setup(device, segmentIdsBuffer, countsBuffer);
    await this.shaders[3].setup(device, meansBufferTranspose, countsBuffer, meansBuffer);
    await this.shaders[4].setup(device, meansBuffer, meansBufferTranspose);
    await this.shaders[5].setup(device, dataBuffer, meansBufferTranspose, segmentIdsBuffer);

    this.isSetup = true;
  }

  encode(pass:GPUComputePassEncoder) {
    if (!this.isSetup) {
      throw new Error('KMeansShader is not setup');
    }
    for(let shader of this.shaders) {
      shader.encode(pass);
    }
  }

  destroy() {
    this.countsBuffer.destroy();
    this.meansBuffer.destroy();
    this.meansBufferTranspose.destroy()
    for(let shader of this.shaders) {
      shader.destroy();
    }
  }

}

async function kmeans(device: GPUDevice, M: number, N: number, K: number, data, segmentIds) {
  const dataBuffer = GPUUtils.createStorageBuffer(device, data);
  const segmentIdsBuffer = GPUUtils.createStorageBuffer(device, segmentIds);

  const shader = new KMeansShader(M, N, K)
  await shader.setup(device, dataBuffer, segmentIdsBuffer);

  for(let i = 0 ; i < 200 ; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    shader.encode(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  const segmentIdsArray = Array.from(await GPUUtils.writeToCPU(device, segmentIdsBuffer, M*4, true));
  const means = await GPUUtils.writeToCPU(device, shader.meansBufferTranspose, K*4*N, false);
  const meansArray = Array.from(new Float32Array(means.buffer));
  return {
    segmentIds: segmentIdsArray,
    means: meansArray
  }

}

export {KMeansShader, kmeans}