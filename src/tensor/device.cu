#include "tensor/device.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace TensorFramework {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

void* allocate_memory(size_t size, const Device& device) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    if (device.is_cpu()) {
        ptr = std::malloc(size);
        if (!ptr) {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
    } else {
        CUDA_CHECK(cudaSetDevice(device.index()));
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }
    return ptr;
}

void free_memory(void* ptr, const Device& device) {
    if (!ptr) return;
    
    if (device.is_cpu()) {
        std::free(ptr);
    } else {
        CUDA_CHECK(cudaSetDevice(device.index()));
        CUDA_CHECK(cudaFree(ptr));
    }
}

void copy_memory(void* dst, const void* src, size_t size,
                 const Device& dst_device, const Device& src_device) {
    if (size == 0) return;
    
    if (src_device.is_cpu() && dst_device.is_cpu()) {
        // CPU to CPU
        std::memcpy(dst, src, size);
    } else if (src_device.is_cpu() && dst_device.is_cuda()) {
        // CPU to CUDA
        CUDA_CHECK(cudaSetDevice(dst_device.index()));
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
        // CUDA to CPU
        CUDA_CHECK(cudaSetDevice(src_device.index()));
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    } else {
        // CUDA to CUDA
        if (src_device.index() == dst_device.index()) {
            CUDA_CHECK(cudaSetDevice(src_device.index()));
            CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyPeer(dst, dst_device.index(), src, src_device.index(), size));
        }
    }
}

} // namespace TensorFramework
