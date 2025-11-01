#include "tensor/tensor.hpp"
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

namespace TF {

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
        return *this;  // 浅拷贝，共享storage
    }
    
    // 创建新的连续tensor并拷贝数据
    Tensor result(shape_, dtype(), device());
    
    // 简化实现：使用CPU端逐元素拷贝
    // 实际应该根据设备类型调用相应的kernel
    
    if (device().is_cpu()) {
        // CPU实现
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
        
        copy_recursive(0, 0, 0);
    } else {
        // CUDA实现：先拷贝到CPU，再拷贝回GPU
        auto cpu_tensor = to(Device(DeviceType::CPU)).contiguous();
        copy_memory(result.data_ptr(), cpu_tensor.data_ptr(),
                   numel() * dtype_size(dtype()),
                   device(), cpu_tensor.device());
    }
    
    return result;
}

Tensor Tensor::to(const Device& device) const {
    if (device == this->device()) {
        return *this;
    }
    
    auto cont = contiguous();
    Tensor result(shape_, dtype(), device);
    copy_memory(result.data_ptr(), cont.data_ptr(),
               numel() * dtype_size(dtype()),
               device, this->device());
    
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

} // namespace TF

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
