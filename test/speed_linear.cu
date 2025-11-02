#include "layers/linear.hpp"
#include "tensor/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace TensorFramework;
using namespace Layers;

// 辅助函数：随机初始化tensor
void random_init(Tensor& tensor, float min_val = -1.0f, float max_val = 1.0f) {
    Tensor cpu_tensor = tensor.to(DeviceType::CPU);
    float* data = static_cast<float*>(cpu_tensor.data_ptr());
    
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        // 简单的伪随机数生成
        data[i] = min_val + (max_val - min_val) * (float)(i % 1000) / 1000.0f;
    }
    
    tensor = cpu_tensor.to(tensor.device());
}

// 计时辅助函数
template<typename Func>
double measure_time_ms(Func func, int warmup = 3, int iterations = 10) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        func();
    }
    
    // 实际测量
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0 / iterations;  // 返回平均毫秒数
}

void speed_linear() {
    std::cout << "\n";
    std::cout << "############################################################" << std::endl;
    std::cout << "#                                                          #" << std::endl;
    std::cout << "#            Linear Layer Speed Benchmark                  #" << std::endl;
    std::cout << "#                                                          #" << std::endl;
    std::cout << "############################################################" << std::endl;
    
    const int64_t batch = 2000;
    const int64_t in_features = 2000;
    const int64_t out_features = 2000;
    const int warmup_iterations = 3;
    const int test_iterations = 10;
    
    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Batch size: " << batch << std::endl;
    std::cout << "Input features: " << in_features << std::endl;
    std::cout << "Output features: " << out_features << std::endl;
    std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
    std::cout << "Test iterations: " << test_iterations << std::endl;
    
    // ==================== CPU 测试 ====================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CPU Performance Test" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer_cpu(in_features, out_features);
    random_init(layer_cpu.weight(), -0.1f, 0.1f);
    random_init(layer_cpu.bias(), -0.01f, 0.01f);
    
    Tensor input_cpu({batch, in_features}, DType::Float32, DeviceType::CPU);
    random_init(input_cpu, -1.0f, 1.0f);
    
    Tensor grad_output_cpu({batch, out_features}, DType::Float32, DeviceType::CPU);
    random_init(grad_output_cpu, -0.1f, 0.1f);
    
    std::cout << "\nPreparing CPU data..." << std::endl;
    
    // Forward pass
    Tensor output_cpu;
    std::cout << "\nTesting CPU Forward Pass..." << std::endl;
    double cpu_forward_time = measure_time_ms([&]() {
        output_cpu = layer_cpu.forward(input_cpu);
    }, warmup_iterations, test_iterations);
    
    // Backward pass
    Tensor grad_input_cpu;
    std::cout << "Testing CPU Backward Pass..." << std::endl;
    double cpu_backward_time = measure_time_ms([&]() {
        grad_input_cpu = layer_cpu.backward(grad_output_cpu);
    }, warmup_iterations, test_iterations);
    
    double cpu_total_time = cpu_forward_time + cpu_backward_time;
    
    std::cout << "\n--- CPU Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Forward:  " << std::setw(10) << cpu_forward_time << " ms" << std::endl;
    std::cout << "Backward: " << std::setw(10) << cpu_backward_time << " ms" << std::endl;
    std::cout << "Total:    " << std::setw(10) << cpu_total_time << " ms" << std::endl;
    
    // ==================== CUDA 测试 ====================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA Performance Test" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer_cuda(in_features, out_features);
    random_init(layer_cuda.weight(), -0.1f, 0.1f);
    random_init(layer_cuda.bias(), -0.01f, 0.01f);
    
    std::cout << "\nPreparing CUDA data..." << std::endl;
    Tensor input_cuda = input_cpu.to(Device(DeviceType::CUDA, 0));
    Tensor grad_output_cuda = grad_output_cpu.to(Device(DeviceType::CUDA, 0));
    
    // Forward pass
    Tensor output_cuda;
    std::cout << "\nTesting CUDA Forward Pass..." << std::endl;
    double cuda_forward_time = measure_time_ms([&]() {
        output_cuda = layer_cuda.forward(input_cuda);
        cudaDeviceSynchronize();  // 确保完成
    }, warmup_iterations, test_iterations);
    
    // Backward pass
    Tensor grad_input_cuda;
    std::cout << "Testing CUDA Backward Pass..." << std::endl;
    double cuda_backward_time = measure_time_ms([&]() {
        grad_input_cuda = layer_cuda.backward(grad_output_cuda);
        cudaDeviceSynchronize();  // 确保完成
    }, warmup_iterations, test_iterations);
    
    double cuda_total_time = cuda_forward_time + cuda_backward_time;
    
    std::cout << "\n--- CUDA Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Forward:  " << std::setw(10) << cuda_forward_time << " ms" << std::endl;
    std::cout << "Backward: " << std::setw(10) << cuda_backward_time << " ms" << std::endl;
    std::cout << "Total:    " << std::setw(10) << cuda_total_time << " ms" << std::endl;
    
    // ==================== 性能对比 ====================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Performance Comparison" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    double forward_speedup = cpu_forward_time / cuda_forward_time;
    double backward_speedup = cpu_backward_time / cuda_backward_time;
    double total_speedup = cpu_total_time / cuda_total_time;
    
    std::cout << "\n--- Speedup (CPU / CUDA) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Forward:  " << std::setw(8) << forward_speedup << "x" << std::endl;
    std::cout << "Backward: " << std::setw(8) << backward_speedup << "x" << std::endl;
    std::cout << "Total:    " << std::setw(8) << total_speedup << "x" << std::endl;
    
    // ==================== 详细统计 ====================
    std::cout << "\n--- Detailed Statistics ---" << std::endl;
    
    // 计算量统计
    int64_t forward_flops = 2LL * batch * in_features * out_features;  // 矩阵乘法
    int64_t backward_input_flops = 2LL * batch * out_features * in_features;  // grad_input
    int64_t backward_weight_flops = 2LL * in_features * batch * out_features;  // grad_weight
    int64_t total_flops = forward_flops + backward_input_flops + backward_weight_flops;
    
    std::cout << "\nComputation:" << std::endl;
    std::cout << "  Forward FLOPs:  " << forward_flops / 1e9 << " GFLOPs" << std::endl;
    std::cout << "  Backward FLOPs: " << (backward_input_flops + backward_weight_flops) / 1e9 << " GFLOPs" << std::endl;
    std::cout << "  Total FLOPs:    " << total_flops / 1e9 << " GFLOPs" << std::endl;
    
    // 吞吐量
    double cpu_throughput = total_flops / (cpu_total_time / 1000.0) / 1e9;  // GFLOPS
    double cuda_throughput = total_flops / (cuda_total_time / 1000.0) / 1e9;
    
    std::cout << "\nThroughput:" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  CPU:  " << std::setw(10) << cpu_throughput << " GFLOPS" << std::endl;
    std::cout << "  CUDA: " << std::setw(10) << cuda_throughput << " GFLOPS" << std::endl;
    
    // 内存使用
    int64_t input_size = batch * in_features * sizeof(float);
    int64_t weight_size = in_features * out_features * sizeof(float);
    int64_t bias_size = out_features * sizeof(float);
    int64_t output_size = batch * out_features * sizeof(float);
    int64_t total_memory = input_size + weight_size + bias_size + output_size;
    
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "  Input:  " << input_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Weight: " << weight_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Bias:   " << bias_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Output: " << output_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Total:  " << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // ==================== 总结 ====================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\n";
    std::cout << "Matrix dimensions: [" << batch << " x " << in_features 
              << "] @ [" << in_features << " x " << out_features << "]" << std::endl;
    std::cout << "\n";
    
    std::cout << std::setw(15) << "Operation" 
              << std::setw(15) << "CPU (ms)" 
              << std::setw(15) << "CUDA (ms)" 
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(15) << "Forward" 
              << std::setw(15) << cpu_forward_time 
              << std::setw(15) << cuda_forward_time 
              << std::setw(14) << forward_speedup << "x" << std::endl;
    
    std::cout << std::setw(15) << "Backward" 
              << std::setw(15) << cpu_backward_time 
              << std::setw(15) << cuda_backward_time 
              << std::setw(14) << backward_speedup << "x" << std::endl;
    
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::setw(15) << "Total" 
              << std::setw(15) << cpu_total_time 
              << std::setw(15) << cuda_total_time 
              << std::setw(14) << total_speedup << "x" << std::endl;
    
    std::cout << "\n";
    std::cout << "############################################################" << std::endl;
    std::cout << "#                                                          #" << std::endl;
    std::cout << "#              Benchmark Completed!                        #" << std::endl;
    std::cout << "#                                                          #" << std::endl;
    std::cout << "############################################################" << std::endl;
    std::cout << "\n";
}

int main() {
    try {
        speed_linear();
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception caught: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
