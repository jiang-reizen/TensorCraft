#pragma once
#include <string>
#include <stdexcept>

namespace framework {

enum class DeviceType {
    CPU,
    CUDA
};

class Device {
public:
    Device(DeviceType type = DeviceType::CPU, int index = 0)
        : type_(type), index_(index) {
        if (type_ == DeviceType::CUDA && index_ < 0) {
            throw std::runtime_error("CUDA device index must be non-negative");
        }
    }

    DeviceType type() const { return type_; }
    int index() const { return index_; }
    bool is_cpu() const { return type_ == DeviceType::CPU; }
    bool is_cuda() const { return type_ == DeviceType::CUDA; }

    std::string str() const {
        if (type_ == DeviceType::CPU) {
            return "cpu";
        } else {
            return "cuda:" + std::to_string(index_);
        }
    }

    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

private:
    DeviceType type_;
    int index_;
};

// 工具函数
void* allocate_memory(size_t size, const Device& device);
void free_memory(void* ptr, const Device& device);
void copy_memory(void* dst, const void* src, size_t size, 
                 const Device& dst_device, const Device& src_device);

} // namespace framework
