// test_contiguous_with_offset.cpp

#include "tensor/tensor.hpp"
#include <iostream>

using namespace TF;

int main() {
    auto x = Tensor::arange(0, 120, 1, DType::Float32, Device(DeviceType::CPU));
    float* cpu_x_data = static_cast<float*>(x.data_ptr());
    std::cout << "First 10 elements of x:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << cpu_x_data[i] << " ";
    }
    std::cout << std::endl;

    // 创建一个大的tensor
    auto t = Tensor::arange(0, 120, 1, DType::Float32, Device(DeviceType::CUDA, 0));
    std::cout << "First 120 elements of t:" << std::endl;
    auto cpu_t = t.to(Device(DeviceType::CPU));
    float* cpu_t_data = static_cast<float*>(cpu_t.data_ptr());
    for (int i = 0; i < 120; ++i) {
        std::cout << cpu_t_data[i] << " ";
    }
    std::cout << std::endl;
    t = t.view({5, 4, 6});
    
    std::cout << "Original tensor: " << t.str() << std::endl;
    
    // 使用slice创建一个有offset的view
    auto t_sliced = t.slice(0, 1, 4, 1);  // shape: [3, 4, 6], offset: 24
    std::cout << "Sliced tensor: " << t_sliced.str() << std::endl;
    std::cout << "Offset: " << t_sliced.offset() << std::endl;
    std::cout << "Is contiguous: " << t_sliced.is_contiguous() << std::endl;
    
    // 再转置
    auto t_trans = t_sliced.transpose(0, 2);  // shape: [6, 4, 3]
    std::cout << "Transposed: " << t_trans.str() << std::endl;
    std::cout << "Is contiguous: " << t_trans.is_contiguous() << std::endl;
    
    // 转为连续（测试offset处理）
    auto t_cont = t_trans.contiguous();
    std::cout << "After contiguous: " << t_cont.str() << std::endl;
    std::cout << "Is contiguous: " << t_cont.is_contiguous() << std::endl;
    
    // 验证数据正确性
    auto cpu_original = t_trans.to(Device(DeviceType::CPU));
    auto cpu_result = t_cont.to(Device(DeviceType::CPU));
    
    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "First 10 elements of original:" << std::endl;
    float* orig_data = static_cast<float*>(cpu_original.data_ptr());
    for (int i = 0; i < 10; ++i) {
        std::cout << orig_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First 10 elements of contiguous:" << std::endl;
    float* cont_data = static_cast<float*>(cpu_result.data_ptr());
    for (int i = 0; i < 10; ++i) {
        std::cout << cont_data[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
