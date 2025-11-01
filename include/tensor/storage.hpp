#pragma once
#include "device.hpp"
#include "dtype.hpp"
#include <memory>
#include <atomic>

namespace framework {

// 引用计数的数据存储
class Storage {
public:
    Storage(size_t size, DType dtype, const Device& device);
    ~Storage();

    // 禁止拷贝，只允许通过智能指针共享
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    size_t nbytes() const { return size_ * dtype_size(dtype_); }
    DType dtype() const { return dtype_; }
    const Device& device() const { return device_; }

    // 创建共享指针
    static std::shared_ptr<Storage> create(size_t size, DType dtype, const Device& device) {
        return std::make_shared<Storage>(size, dtype, device);
    }

private:
    void* data_;
    size_t size_;  // 元素数量
    DType dtype_;
    Device device_;
};

} // namespace framework
