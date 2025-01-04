export interface ShaderEncoder {
    encode(pass:GPUComputePassEncoder):void;
    setup(...args: any[]): Promise<void>;
}