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

export function createStorageBuffer(device: GPUDevice, array) {
    const buffer = device.createBuffer( 
        {
            size: array.byteLength,
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

}

export {GPUUtils};