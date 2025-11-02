#pragma once
#include <string>
#include <stdexcept>

// Tensor Framewrok
namespace TensorFramework {

// Device Type 分为 CPU 和 CUDA，其中 CUDA 还有编号
enum class DeviceType {
    CPU,
    CUDA
};

class Device {
private:
    DeviceType type_;
    int index_;
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


};

// 工具函数，根据 Device 类型分配空间。
// size 为字节数。
void* allocate_memory(size_t size, const Device& device);
// 工具函数，根据 Device 类型释放空间
void free_memory(void* ptr, const Device& device);
// 工具函数，根据 Device 类型拷贝数据。
// size 为字节数。
void copy_memory(void* dst, const void* src, size_t size, 
                 const Device& dst_device, const Device& src_device);

} // namespace TensorFramework
