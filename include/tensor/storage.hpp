#pragma once
#include "tensor/device.hpp"
#include "tensor/dtype.hpp"
#include <memory>
#include <atomic>

// Tensor Framework
namespace TensorFramework {

// 引用计数的数据存储
class Storage {
private:
    void* data_;
    size_t size_;  // 元素数量
    DType dtype_;
    Device device_;
public:
    // 分配空间
    Storage(size_t size, DType dtype, const Device& device);
    // 释放空间
    ~Storage();

    // 禁止拷贝，只允许通过智能指针共享
    Storage(const Storage&) = delete;
    // 禁止拷贝，只允许通过智能指针共享
    Storage& operator=(const Storage&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    // 使用字节数
    size_t nbytes() const { return size_ * dtype_size(dtype_); }
    DType dtype() const { return dtype_; }
    const Device& device() const { return device_; }

    // 创建共享指针
    static std::shared_ptr<Storage> create(size_t size, DType dtype, const Device& device) {
        return std::make_shared<Storage>(size, dtype, device);
    }

};

} // namespace TensorFramework
