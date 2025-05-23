namespace GPUUtils {

export function createUniform(device: GPUDevice, array) {
    const buffer = device.createBuffer( 
        {
            size: array.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
        }
    );

    device.queue.writeBuffer(buffer, 0, array, 0, array.length)
    return buffer
}

function nextPowerOf2(n: number): number {
    if (n <= 1) {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    // If n can exceed 32 bits (e.g., in environments with 64-bit numbers):
    // n |= n >> 32; 
    return n + 1;
}

export function createStorageBuffer(device: GPUDevice, array, padded=false) {
    const sizeof = array.byteLength / array.length;
    let size = array.byteLength;
    if (padded) {
        size = sizeof*nextPowerOf2(array.length);
    }

    const buffer = device.createBuffer( 
        {
            size: size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC 
        }
    );

    device.queue.writeBuffer(buffer, 0, array, 0, array.length)
    return buffer
}


export function createShaderModule(device: GPUDevice, code: string, name: string, workgroupSize: Array<number>) {
    const wgsl = code.replace("${workgroupSize}", workgroupSize.join(", "));
    return device.createShaderModule({
        label: code,
        code: wgsl
    });
}

export function createBindGroup(device: GPUDevice, layout: GPUBindGroupLayout, bindings: Array<GPUBuffer>) {
    const entries = bindings.map((binding, index) => {
        return {
            binding: index,
            resource: {
            buffer: binding
            }
        } as GPUBindGroupEntry;

    });
    return device.createBindGroup(
        { 
            entries: entries,
            layout: layout
        }
    );
}

export async function writeToCPU(device: GPUDevice, buffer: GPUBuffer, size: number, uint: boolean = true) {
    const stagingBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = stagingBuffer.getMappedRange();
    var data;
    if (uint) {
        data = new Uint32Array(copyArrayBuffer.slice(0));
    } else {
        data = new Float32Array(copyArrayBuffer.slice(0));
    }
    
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    return data
}

export async function log(device:GPUDevice, buffer:GPUBuffer, uint: boolean) {
    let arr = await writeToCPU(device, buffer, buffer.size, uint);
    console.log(arr)
}

}

export {GPUUtils};