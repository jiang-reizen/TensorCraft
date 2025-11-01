#include "tensor/storage.hpp"
#include <stdexcept>

namespace framework {

Storage::Storage(size_t size, DType dtype, const Device& device)
    : size_(size), dtype_(dtype), device_(device) {
    if (size > 0) {
        data_ = allocate_memory(size * dtype_size(dtype), device);
    } else {
        data_ = nullptr;
    }
}

Storage::~Storage() {
    if (data_) {
        free_memory(data_, device_);
    }
}

} // namespace framework
