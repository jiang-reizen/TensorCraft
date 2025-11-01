#include "tensor/tensor.hpp"
#include <iostream>
#include <cassert>

using namespace framework;

void test_basic_creation() {
    std::cout << "Testing basic tensor creation..." << std::endl;
    
    auto t1 = Tensor::zeros({2, 3}, DType::Float32, Device(DeviceType::CPU));
    assert(t1.shape()[0] == 2);
    assert(t1.shape()[1] == 3);
    assert(t1.numel() == 6);
    
    std::cout << "[Pass] Basic creation test passed" << std::endl;
}

void test_view_operations() {
    std::cout << "Testing view operations..." << std::endl;
    
    auto t1 = Tensor::ones({2, 3, 4}, DType::Float32, Device(DeviceType::CPU));
    
    // Test view
    auto t2 = t1.view({2, 12});
    assert(t2.shape()[0] == 2);
    assert(t2.shape()[1] == 12);
    assert(t2.numel() == 24);
    
    // Test transpose
    auto t3 = t1.transpose(0, 1);
    assert(t3.shape()[0] == 3);
    assert(t3.shape()[1] == 2);
    assert(!t3.is_contiguous());
    
    std::cout << "[Pass] View operations test passed" << std::endl;
}

void test_memory_management() {
    std::cout << "Testing memory management..." << std::endl;
    
    auto t1 = Tensor::randn({100, 100}, DType::Float32, Device(DeviceType::CPU));
    
    // Shallow copy (view)
    auto t2 = t1.view({10000});
    
    // Deep copy
    auto t3 = t1.clone();
    
    std::cout << "[Pass] Memory management test passed" << std::endl;
}

#ifdef CUDA_AVAILABLE
void test_cuda_operations() {
    std::cout << "Testing CUDA operations..." << std::endl;
    
    auto t_cpu = Tensor::ones({10, 10}, DType::Float32, Device(DeviceType::CPU));
    
    // Transfer to GPU
    auto t_gpu = t_cpu.to(Device(DeviceType::CUDA, 0));
    assert(t_gpu.device().is_cuda());
    
    // Transfer back to CPU
    auto t_cpu2 = t_gpu.to(Device(DeviceType::CPU));
    assert(t_cpu2.device().is_cpu());
    
    std::cout << "[Pass] CUDA operations test passed" << std::endl;
}
#endif

int main() {
    std::cout << "=== Running Tensor Tests ===" << std::endl << std::endl;
    
    try {
        test_basic_creation();
        test_view_operations();
        test_memory_management();
        
#ifdef CUDA_AVAILABLE
        test_cuda_operations();
#else
        std::cout << "[Warning] CUDA tests skipped (CUDA not available)" << std::endl;
#endif
        
        std::cout << std::endl << "=== All Tests Passed ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[Fail] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
