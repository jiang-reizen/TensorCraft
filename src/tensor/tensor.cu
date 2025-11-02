#include "tensor/tensor.hpp"
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

namespace TensorFramework {

// ============ 构造函数 ============

Tensor::Tensor() 
    : storage_(nullptr), shape_({}), stride_({}), offset_(0) {}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, const Device& device)
    : shape_(shape), offset_(0) {
    stride_ = compute_stride(shape);
    int64_t size = numel();
    storage_ = Storage::create(size, dtype, device);
}

Tensor::Tensor(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               const std::vector<int64_t>& stride,
               int64_t offset)
    : storage_(storage), shape_(shape), stride_(stride), offset_(offset) {}

// ============ 辅助函数 ============

std::vector<int64_t> Tensor::compute_stride(const std::vector<int64_t>& shape) {
    std::vector<int64_t> stride(shape.size());
    if (shape.empty()) return stride;
    
    stride.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
}

int64_t Tensor::compute_offset(const std::vector<int64_t>& indices) const {
    int64_t offset = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * stride_[i];
    }
    return offset;
}

void Tensor::check_device_match(const Tensor& other) const {
    if (device() != other.device()) {
        throw std::runtime_error("Tensors must be on the same device");
    }
}

void Tensor::check_shape_match(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch");
    }
}


// ============ 基本函数 ============

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

bool Tensor::is_contiguous() const {
    auto expected_stride = compute_stride(shape_);
    return stride_ == expected_stride;
}

// ============ 内存管理 ============

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype(), device());
    
    if (is_contiguous()) {
        copy_memory(result.data_ptr(), data_ptr(), numel() * dtype_size(dtype()),
                   device(), device());
    } else {
        // 对于非连续tensor，需要逐元素拷贝
        auto cont = contiguous();
        copy_memory(result.data_ptr(), cont.data_ptr(), numel() * dtype_size(dtype()),
                   device(), device());
    }
    
    return result;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    
    Tensor result(shape_, dtype(), device());
    
    if (device().is_cpu()) {
        // CPU实现（保持原有逻辑）
        std::function<void(int64_t, int64_t, int64_t)> copy_recursive;
        copy_recursive = [&](int64_t dim, int64_t src_offset, int64_t dst_offset) {
            if (dim == ndim()) {
                std::memcpy(
                    static_cast<char*>(result.data_ptr()) + dst_offset * dtype_size(dtype()),
                    static_cast<const char*>(data_ptr()) + src_offset * dtype_size(dtype()),
                    dtype_size(dtype())
                );
                return;
            }
            
            for (int64_t i = 0; i < shape_[dim]; ++i) {
                copy_recursive(dim + 1,
                             src_offset + i * stride_[dim],
                             dst_offset + i * result.stride_[dim]);
            }
        };
        
        copy_recursive(0, offset_, 0);  // 注意这里传入offset_
    } else {
        // CUDA实现：调用GPU kernel，传入offset
        copy_strided_to_contiguous_cuda(
            result.data_ptr(),
            storage_->data(),  // 使用storage的原始指针
            shape_,
            stride_,
            offset_,           // 传入offset
            ndim(),
            numel(),
            dtype_size(dtype())
        );
        cudaSetDevice(device().index());
        cudaDeviceSynchronize();
    }
    
    return result;
}

Tensor Tensor::to(const Device& device) const {
    if (device == this->device()) {
        return *this;
    }
    if (this->device().is_cuda()) {
        cudaSetDevice(this->device().index());
        cudaDeviceSynchronize();
    }
    auto cont = contiguous();
    Tensor result(shape_, dtype(), device);
    copy_memory(result.data_ptr(), cont.data_ptr(),
               numel() * dtype_size(dtype()),
               device, this->device());
    if (device.is_cuda()) {
        cudaSetDevice(device.index());
        cudaDeviceSynchronize();
    }
    
    return result;
}

// ============ 视图操作 ============

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    int64_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1LL, std::multiplies<int64_t>());
    if (new_numel != numel()) {
        throw std::runtime_error("View size mismatch");
    }
    
    if (!is_contiguous()) {
        throw std::runtime_error("View requires contiguous tensor");
    }
    
    return Tensor(storage_, new_shape, compute_stride(new_shape), offset_);
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    if (is_contiguous()) {
        return view(new_shape);
    } else {
        return contiguous().view(new_shape);
    }
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0 || dim0 >= ndim()) throw std::runtime_error("[Error] [Transpose] dim0 invalid");
    if (dim1 < 0 || dim1 >= ndim()) throw std::runtime_error("[Error] [Transpose] dim1 invalid");
    
    auto new_shape = shape_;
    auto new_stride = stride_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_stride[dim0], new_stride[dim1]);
    
    return Tensor(storage_, new_shape, new_stride, offset_);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    if (dim < 0 || dim > ndim()) throw std::runtime_error("[Error] [Slice] dim invalid");
    if (start < 0 || start >= shape_[dim]) throw std::runtime_error("[Error] [Slice] start invalid");
    if (end < 0 || end >= shape_[dim]) throw std::runtime_error("[Error] [Slice] end invalid");
    if (start >= end) throw std::runtime_error("[Error] [Slice] interval invalid");
    
    auto new_shape = shape_;
    auto new_stride = stride_;
    int64_t new_offset = offset_ + start * stride_[dim];
    
    new_shape[dim] = (end - start + step - 1) / step;
    new_stride[dim] *= step;
    
    return Tensor(storage_, new_shape, new_stride, new_offset);
}

Tensor Tensor::operator[](int64_t index) const {
    if (ndim() == 0) {
        throw std::runtime_error("[Error] [operator[]] Cannot index 0-dim tensor");
    }
    
    if (index < 0 || index >= shape_[0]) {
        throw std::runtime_error("[Error] [operator[]] Index out of range");
    }
    
    std::vector<int64_t> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<int64_t> new_stride(stride_.begin() + 1, stride_.end());
    int64_t new_offset = offset_ + index * stride_[0];
    
    return Tensor(storage_, new_shape, new_stride, new_offset);
}
/*
// ============ 工厂方法 ============

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    result.zero_();
    return result;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    result.ones_();
    return result;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    if (device.is_cuda()) {
        throw std::runtime_error("randn on CUDA not implemented yet");
    }
    
    Tensor result(shape, dtype, device);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    float* data = static_cast<float*>(result.data_ptr());
    for (int64_t i = 0; i < result.numel(); ++i) {
        data[i] = dist(gen);
    }
    
    return result;
}

Tensor Tensor::arange(float start, float end, float step, DType dtype, const Device& device) {
    int64_t size = static_cast<int64_t>(std::ceil((end - start) / step));
    Tensor result({size}, dtype, device);
    
    if (device.is_cpu()) {
        float* data = static_cast<float*>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = start + i * step;
        }
    } else {
        // CUDA实现省略
        throw std::runtime_error("arange on CUDA not implemented yet");
    }
    
    return result;
}

// ============ 数据填充 ============

void Tensor::zero_() {
    if (device().is_cpu()) {
        std::memset(data_ptr(), 0, numel() * dtype_size(dtype()));
    } else {
        cudaMemset(data_ptr(), 0, numel() * dtype_size(dtype()));
    }
}

void Tensor::ones_() {
    fill_(1.0f);
}

void Tensor::fill_(float value) {
    if (!device().is_cpu()) {
        throw std::runtime_error("fill_ on CUDA not implemented yet");
    }
    
    // 简化实现，仅支持float32
    if (dtype() != DType::Float32) {
        throw std::runtime_error("fill_ only supports float32 currently");
    }
    
    float* data = static_cast<float*>(data_ptr());
    for (int64_t i = 0; i < numel(); ++i) {
        data[i] = value;
    }
}
*/

// ============ 调试 ============

void Tensor::print_basic_info() const {
    std::cout << str() << std::endl;
}

void Tensor::print() const {
    std::cout << str() << std::endl;
    Tensor temp = clone();
    temp = temp.to(DeviceType::CPU);
    for (int i = 0; i < temp.numel(); ++i) {
        if (dtype_size(dtype()) == 8) 
            std::cout << *(uint64_t*)(static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype())) << std::endl;
        else if (dtype_size(dtype()) == 4)
            std::cout << *(uint32_t*)(static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype())) << std::endl;
        else if (dtype_size(dtype()) == 2)
            std::cout << *(uint16_t*)(static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype())) << std::endl;
    }
}

std::string Tensor::str() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i < shape_.size() - 1) oss << ", ";
    }
    oss << "], dtype=" << dtype_name(dtype());
    oss << ", device=" << device().str();
    oss << ", contiguous=" << (is_contiguous() ? "True" : "False");
    oss << ")";
    return oss.str();
}

} // namespace TensorFramework

namespace TensorFramework {

// 将线性索引转换为多维索引
__device__ void linear_to_multi_index(
    int64_t linear_idx,
    const int64_t* shape,
    int64_t ndim,
    int64_t* multi_idx
) {
    for (int64_t i = ndim - 1; i >= 0; --i) {
        multi_idx[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
}

// 根据多维索引和stride计算在strided tensor中的偏移
__device__ int64_t multi_index_to_offset(
    const int64_t* multi_idx,
    const int64_t* stride,
    int64_t ndim,
    int64_t base_offset  // 基础偏移
) {
    int64_t offset = base_offset;
    for (int64_t i = 0; i < ndim; ++i) {
        offset += multi_idx[i] * stride[i];
    }
    return offset;
}

// 通用的 strided copy kernel（模板版本，支持不同数据类型）
template<typename T>
__global__ void strided_copy_kernel(
    T* dst,                    // 连续输出
    const T* src,              // strided输入
    const int64_t* shape,      // 形状
    const int64_t* stride,     // 步长
    int64_t base_offset,       // 基础偏移
    int64_t ndim,              // 维度数
    int64_t numel              // 总元素数
) {
    // 全局线程索引
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numel) return;
    
    // 为每个线程分配栈上的多维索引数组
    // 注意：CUDA不支持动态栈数组，这里假设最大维度为8
    int64_t multi_idx[8];  // 假设最大8维
    
    // if (ndim > 8) {
    //     // 对于超过8维的情况，需要使用其他方法
    //     return;
    // }
    
    // 将线性索引转换为多维索引
    linear_to_multi_index(idx, shape, ndim, multi_idx);
    
    // 计算在strided tensor中的偏移（包含base_offset）
    int64_t src_offset = multi_index_to_offset(multi_idx, stride, ndim, base_offset);
    
    // 拷贝数据
    dst[idx] = src[src_offset];
}

void copy_strided_to_contiguous_cuda(
    void* dst,
    const void* src,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    int64_t offset,
    int64_t ndim,
    int64_t numel,
    size_t element_size
) {
    if (numel == 0) return;
    
    // 配置kernel启动参数
    const int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    
    // 根据维度和元素大小选择合适的kernel
    // 优化：针对常见维度使用特化版本

    // 通用版本：将shape和stride拷贝到GPU
    int64_t* d_shape = nullptr;
    int64_t* d_stride = nullptr;
    
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_stride, ndim * sizeof(int64_t));
    
    cudaMemcpy(d_shape, shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, stride.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    // 根据元素大小选择kernel
    switch (element_size) {
        case 1:  // int8, uint8
            strided_copy_kernel<uint8_t><<<grid_size, block_size>>>(
                static_cast<uint8_t*>(dst),
                static_cast<const uint8_t*>(src),
                d_shape, d_stride, offset, ndim, numel
            );
            break;
        case 2:  // float16, int16
            strided_copy_kernel<uint16_t><<<grid_size, block_size>>>(
                static_cast<uint16_t*>(dst),
                static_cast<const uint16_t*>(src),
                d_shape, d_stride, offset, ndim, numel
            );
            break;
        case 4:  // float32, int32
            strided_copy_kernel<uint32_t><<<grid_size, block_size>>>(
                static_cast<uint32_t*>(dst),
                static_cast<const uint32_t*>(src),
                d_shape, d_stride, offset, ndim, numel
            );
            break;
        case 8:  // float64, int64
            strided_copy_kernel<uint64_t><<<grid_size, block_size>>>(
                static_cast<uint64_t*>(dst),
                static_cast<const uint64_t*>(src),
                d_shape, d_stride, offset, ndim, numel
            );
            break;
        default:
            cudaFree(d_shape);
            cudaFree(d_stride);
            throw std::runtime_error("Unsupported element size");
    }
    
    // 清理
    cudaFree(d_shape);
    cudaFree(d_stride);
    
    // 检查kernel执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
    }
    
    // 同步（可选，根据需要决定是否异步）
    cudaDeviceSynchronize();
}


} // namespace TensorFramework

namespace TensorFramework {

__global__ void arange_kernel(float* dst, float start, float step, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = start + step * idx;
    }
}

Tensor Tensor::arange(float start, float end, float step, DType dtype, const Device& device) {
    // 仅实现了 dtype == Float32 的逻辑
    int64_t size = static_cast<int64_t>(std::ceil((end - start) / step));
    Tensor result({size}, dtype, device);
    
    if (device.is_cpu()) {
        float* data = static_cast<float*>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = start + i * step;
        }
    } else {
        cudaSetDevice(device.index());
        
        const int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        float* data = static_cast<float*>(result.data_ptr());
        
        // 启动 kernel
        arange_kernel<<<grid_size, block_size>>>(data, start, step, size);
        
        // ✅ 检查 kernel 启动错误
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            throw std::runtime_error(
                std::string("arange_kernel launch failed: ") + 
                cudaGetErrorString(launch_err)
            );
        }
        
        // ✅ 同步等待 kernel 完成
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            throw std::runtime_error(
                std::string("arange_kernel sync failed: ") + 
                cudaGetErrorString(sync_err)
            );
        }
    }
    
    return result;
}

} // namespace TensorFramework

// 继续 tensor.cu

// namespace framework {

// // ============ 基本运算 ============
// // 注意：这里只提供框架，实际的CUDA kernel需要另外实现

// Tensor Tensor::operator+(const Tensor& other) const {
//     check_device_match(other);
//     check_shape_match(other);
    
//     Tensor result(shape_, dtype(), device());
    
//     // 调用相应的kernel或CPU函数
//     // add_kernel(result.data_ptr(), data_ptr(), other.data_ptr(), numel(), device());
    
//     return result;
// }

// Tensor Tensor::operator*(const Tensor& other) const {
//     check_device_match(other);
//     check_shape_match(other);
    
//     Tensor result(shape_, dtype(), device());
//     // mul_kernel(result.data_ptr(), data_ptr(), other.data_ptr(), numel(), device());
    
//     return result;
// }

// Tensor Tensor::matmul(const Tensor& other) const {
//     if (ndim() < 2 || other.ndim() < 2) {
//         throw std::runtime_error("matmul requires at least 2D tensors");
//     }
    
//     int64_t m = shape_[ndim() - 2];
//     int64_t k = shape_[ndim() - 1];
//     int64_t n = other.shape_[other.ndim() - 1];
    
//     if (k != other.shape_[other.ndim() - 2]) {
//         throw std::runtime_error("matmul dimension mismatch");
//     }
    
//     std::vector<int64_t> result_shape = shape_;
//     result_shape[ndim() - 1] = n;
    
//     Tensor result(result_shape, dtype(), device());
    
//     // 调用BLAS或自定义matmul kernel
//     // matmul_kernel(result.data_ptr(), data_ptr(), other.data_ptr(), m, n, k, device());
    
//     return result;
// }

// Tensor Tensor::relu() const {
//     Tensor result(shape_, dtype(), device());
//     // relu_kernel(result.data_ptr(), data_ptr(), numel(), device());
//     return result;
// }

// Tensor Tensor::softmax(int64_t dim) const {
//     if (dim < 0) dim += ndim();
    
//     Tensor result(shape_, dtype(), device());
//     // softmax_kernel(result.data_ptr(), data_ptr(), shape_, dim, device());
//     return result;
// }

// } // namespace framework
