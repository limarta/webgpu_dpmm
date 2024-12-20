import ThreeFryCode from '../shaders/distributions/threefry.wgsl';
import UniformCode from '../shaders/distributions/uniform.wgsl';
import BoxMullerCode from '../shaders/distributions/boxmuller.wgsl';
import { GPUUtils } from "./gpu";

namespace Random {
export async function prng(device: GPUDevice, seed: Uint32Array, N: number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*4*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        rngBuffer, 0,
        cpuBuffer, 0,
        N*4*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Uint32Array(cpuBuffer.getMappedRange())
    const result = new Uint32Array(ans);
    cpuBuffer.unmap();
    return result;
}

export async function threefry(device: GPUDevice, seedBuffer:GPUBuffer, nBuffer:GPUBuffer, rngBuffer:GPUBuffer, N: number) {
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: seedBuffer 
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: nBuffer 
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: rngBuffer 
                }
            }
        ]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        code: ThreeFryCode
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "threefry",
        }
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass()

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    let N_workgroups = Math.ceil(N/16);
    pass.dispatchWorkgroups(N_workgroups,1,1);

    pass.end();
    const commands = encoder.finish();

    device.queue.submit([commands]);
}

async function boxmuller(device:GPUDevice, rngBuffer:GPUBuffer, nBuffer:GPUBuffer, samplesBuffer:GPUBuffer, N:number) {
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: nBuffer 
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: rngBuffer 
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: samplesBuffer 
                }
            }
        ]
    });
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        code: BoxMullerCode
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "boxmuller",
        }
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass()

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    let N_workgroups = Math.ceil(N/16);
    pass.dispatchWorkgroups(N_workgroups,1,1);

    pass.end();
    const commands = encoder.finish();

    device.queue.submit([commands]);
}

export async function normal(device: GPUDevice, seed: Uint32Array, N: number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    const outputBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*2));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    
    boxmuller(device, rngBuffer, nBuffer, outputBuffer, N)

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*2*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        outputBuffer, 0,
        cpuBuffer, 0,
        N*2*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Float32Array(cpuBuffer.getMappedRange())
    const result = new Float32Array(ans);
    console.log(result)
    cpuBuffer.unmap();
    return result;
}

async function uniform_(device:GPUDevice, rngBuffer:GPUBuffer, nBuffer:GPUBuffer, samplesBuffer:GPUBuffer, N:number) {
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: nBuffer 
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: rngBuffer 
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: samplesBuffer 
                }
            }
        ]
    });
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        code: UniformCode
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "uni",
        }
    });

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass()

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    let N_workgroups = Math.ceil(N/16);
    pass.dispatchWorkgroups(N_workgroups,1,1);

    pass.end();
    const commands = encoder.finish();

    device.queue.submit([commands]);
}

export async function uniform(device:GPUDevice, seed: Uint32Array, N:number) {
    const seedBuffer = GPUUtils.createUniform(device, seed);
    const nBuffer = GPUUtils.createUniform(device, new Uint32Array([N]));
    const rngBuffer = GPUUtils.createStorageBuffer(device, new Uint32Array(N*4));
    const samplesBuffer = GPUUtils.createStorageBuffer(device, new Float32Array(N*4));
    threefry(device, seedBuffer, nBuffer, rngBuffer, N);    
    uniform_(device, rngBuffer, nBuffer, samplesBuffer, N);

    const encoder = device.createCommandEncoder();

    const cpuBuffer = device.createBuffer({
        size: N*4*4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(
        samplesBuffer, 0,
        cpuBuffer, 0,
        N*4*4
    );

    const commands = encoder.finish();

    device.queue.submit([commands]);

    await cpuBuffer.mapAsync(GPUMapMode.READ);
    const ans = new Float32Array(cpuBuffer.getMappedRange())
    const result = new Float32Array(ans);
    cpuBuffer.unmap();
    cpuBuffer.destroy();
    return result;
}
}

export {Random};