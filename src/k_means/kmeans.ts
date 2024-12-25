import {Random} from '../utils/rng.ts';
import {GPUUtils} from '../utils/gpu.ts';

namespace DPMM{

export class Numericals {
  seedBuffer: GPUBuffer

  muDimsBuffer: GPUBuffer
  muRandBuffer: GPUBuffer
  muBuffer: GPUBuffer
  dataBuffer: GPUBuffer
  scratchBuffer: GPUBuffer
  N: number // Number of samples
  F: number // Number of numerical featuers
  K: number // Number of clusters
  K_max: number // Maximum possible number of clusters on buffer

  muRandShader: Random.NormalShaderEncoder
  constructor(N: number, F: number, K:number, K_max: number) {
    if (K > K_max) {
        throw new Error(`K=${K} is greater than K_max=${K_max}`);
    }
    this.seedBuffer = null;
    this.muBuffer = null;
    this.muBuffer = null;
    this.muDimsBuffer = null;
    this.scratchBuffer = null;
    this.N = N;
    this.F = F;
    this.K = K;
    this.K_max = K_max;
  }

  async setup(device, seedBuffer: GPUBuffer, dataBuffer: GPUBuffer) {
    this.seedBuffer = seedBuffer;
    this.dataBuffer = dataBuffer;
    this.muDimsBuffer = GPUUtils.createUniform(device, new Uint32Array([this.K_max*this.F]));
    this.muRandBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(this.K_max*this.F*4));
    this.muBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(this.K_max*this.F));

    this.muRandShader = new Random.NormalShaderEncoder(
        device, 
        this.K_max*this.F,
        this.seedBuffer, 
        this.muDimsBuffer, 
        this.muRandBuffer, 
        this.muBuffer
    );

    const UpdateMeanBindGroupLayout = device.createBindGroupLayout({
        label: "RefineMean BGL",
        entries: [
            {
                binding: 0, 
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'uniform'
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage'
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage'
                }
            }
        ]
    });
  }

  encode_init(pass:GPUComputePassEncoder) {
    this.muRandShader.encode(pass);
  }

  encode_update_means(encoder:GPURenderPassEncoder) {

  }

  encode_update_assignments(encoder:GPURenderPassEncoder) {

  }
}

// type Assignments = {
//   assignments: GPUBuffer
// }

// type ClusterProportions = {

// }

export class KMeans {
    device: GPUDevice
    N: number
    F: number
    K: number
    data: Float32Array
    numericals: Numericals
    constructor(K: number) {
        this.N = 16;
        this.K = K;
        this.F = 1;
        this.numericals = new Numericals(this.N, this.F, this.K, this.K)
    }

    async setup(device: GPUDevice) {
        this.device = device
        const seedBuffer = GPUUtils.createUniform(device, new Uint32Array([0,0,0,1]))
        const dataBuffer = GPUUtils.createStorageBuffer(device, new Float32Array([0.0, 0.0, 0.0, 0.0]))
        await this.numericals.setup(device, seedBuffer, dataBuffer)

    }

    async step(device:GPUDevice) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        this.numericals.encode_init(pass);
        pass.end();

        device.queue.submit([encoder.finish()]);
    }
}
// type DPMM = {
//   N: number
//   pi: ClusterProportions,
//   numericals: Numericals,
//   assignments: Assignments
// }

}

export {DPMM};