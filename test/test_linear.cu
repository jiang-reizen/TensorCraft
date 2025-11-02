#include "layers/linear.hpp"
#include "tensor/tensor.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace TensorFramework;
using namespace Layers;

// ============ 辅助函数 ============

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << "[" << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m") << "] " << test_name << std::endl;
}

void print_tensor(const std::string& name, const Tensor& tensor, int max_elements = 20) {
    Tensor cpu_tensor = tensor.to(DeviceType::CPU);
    float* data = static_cast<float*>(cpu_tensor.data_ptr());
    
    std::cout << name << " (shape: [";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        std::cout << tensor.shape()[i];
        if (i < tensor.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]):" << std::endl;
    
    int64_t total = tensor.numel();
    int64_t to_print = std::min(total, (int64_t)max_elements);
    
    std::cout << "  [";
    for (int64_t i = 0; i < to_print; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i];
        if (i < to_print - 1) std::cout << ", ";
        if ((i + 1) % 10 == 0 && i < to_print - 1) std::cout << "\n   ";
    }
    if (total > to_print) {
        std::cout << ", ... (+" << (total - to_print) << " more)";
    }
    std::cout << "]" << std::endl;
}

void print_matrix(const std::string& name, const Tensor& tensor, int max_rows = 10, int max_cols = 10) {
    if (tensor.shape().size() != 2) {
        print_tensor(name, tensor);
        return;
    }
    
    Tensor cpu_tensor = tensor.to(DeviceType::CPU);
    float* data = static_cast<float*>(cpu_tensor.data_ptr());
    
    int64_t rows = tensor.shape()[0];
    int64_t cols = tensor.shape()[1];
    int64_t show_rows = std::min(rows, (int64_t)max_rows);
    int64_t show_cols = std::min(cols, (int64_t)max_cols);
    
    std::cout << name << " [" << rows << " x " << cols << "]:" << std::endl;
    
    for (int64_t i = 0; i < show_rows; ++i) {
        std::cout << "  [";
        for (int64_t j = 0; j < show_cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << data[i * cols + j];
            if (j < show_cols - 1) std::cout << ", ";
        }
        if (cols > show_cols) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (rows > show_rows) {
        std::cout << "  ... (+" << (rows - show_rows) << " more rows)" << std::endl;
    }
}

bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-5, float atol = 1e-8) {
    if (a.shape() != b.shape()) {
        std::cout << "  Shape mismatch!" << std::endl;
        return false;
    }
    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
        std::cout << "  DType mismatch!" << std::endl;
        return false;
    }
    
    Tensor a_cpu = a.to(DeviceType::CPU);
    Tensor b_cpu = b.to(DeviceType::CPU);
    
    float* a_data = static_cast<float*>(a_cpu.data_ptr());
    float* b_data = static_cast<float*>(b_cpu.data_ptr());
    
    int mismatch_count = 0;
    for (int64_t i = 0; i < a.numel(); ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            if (mismatch_count < 5) {  // 只打印前5个不匹配
                std::cout << "  Mismatch at index " << i << ": " 
                          << a_data[i] << " vs " << b_data[i] 
                          << " (diff: " << diff << ", threshold: " << threshold << ")" << std::endl;
            }
            mismatch_count++;
        }
    }
    
    if (mismatch_count > 5) {
        std::cout << "  ... and " << (mismatch_count - 5) << " more mismatches" << std::endl;
    }
    
    return mismatch_count == 0;
}

void init_weights(Tensor& weight, Tensor& bias, float w_val = 0.5f, float b_val = 0.1f) {
    Tensor w_cpu = weight.to(DeviceType::CPU);
    Tensor b_cpu = bias.to(DeviceType::CPU);
    
    float* w_data = static_cast<float*>(w_cpu.data_ptr());
    float* b_data = static_cast<float*>(b_cpu.data_ptr());
    
    for (int64_t i = 0; i < weight.numel(); ++i) {
        w_data[i] = w_val;
    }
    for (int64_t i = 0; i < bias.numel(); ++i) {
        b_data[i] = b_val;
    }
    
    weight = w_cpu.to(weight.device());
    bias = b_cpu.to(bias.device());
}

// ============ 测试函数 ============

// 测试1：基本前向传播 (CPU)
void test_forward_cpu_detailed() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: Forward Pass (CPU) - Detailed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer(3, 2);
    init_weights(layer.weight(), layer.bias(), 0.5f, 0.1f);
    
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    std::cout << "Input features: 3, Output features: 2" << std::endl;
    print_matrix("Weight", layer.weight());
    print_tensor("Bias", layer.bias());
    
    Tensor input({2, 3}, DType::Float32, DeviceType::CPU);
    float* input_data = static_cast<float*>(input.data_ptr());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f;
    input_data[3] = 4.0f; input_data[4] = 5.0f; input_data[5] = 6.0f;
    
    std::cout << "\n--- Input ---" << std::endl;
    print_matrix("Input", input);
    
    Tensor output = layer.forward(input);
    
    std::cout << "\n--- Output ---" << std::endl;
    print_matrix("Output", output);
    
    std::cout << "\n--- Expected Calculation ---" << std::endl;
    std::cout << "output[0,0] = (1.0*0.5 + 2.0*0.5 + 3.0*0.5) + 0.1 = 3.1" << std::endl;
    std::cout << "output[0,1] = (1.0*0.5 + 2.0*0.5 + 3.0*0.5) + 0.1 = 3.1" << std::endl;
    std::cout << "output[1,0] = (4.0*0.5 + 5.0*0.5 + 6.0*0.5) + 0.1 = 7.6" << std::endl;
    std::cout << "output[1,1] = (4.0*0.5 + 5.0*0.5 + 6.0*0.5) + 0.1 = 7.6" << std::endl;
    
    bool shape_ok = (output.shape() == std::vector<int64_t>{2, 2});
    float* output_data = static_cast<float*>(output.data_ptr());
    bool values_ok = (std::abs(output_data[0] - 3.1f) < 1e-5 &&
                      std::abs(output_data[1] - 3.1f) < 1e-5 &&
                      std::abs(output_data[2] - 7.6f) < 1e-5 &&
                      std::abs(output_data[3] - 7.6f) < 1e-5);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Forward CPU - Shape", shape_ok);
    print_test_result("Forward CPU - Values", values_ok);
}

// 测试2：反向传播 (CPU)
void test_backward_cpu_detailed() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: Backward Pass (CPU) - Detailed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer(3, 2);
    init_weights(layer.weight(), layer.bias(), 0.5f, 0.1f);
    
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    print_matrix("Weight", layer.weight());
    print_tensor("Bias", layer.bias());
    
    Tensor input({2, 3}, DType::Float32, DeviceType::CPU);
    float* input_data = static_cast<float*>(input.data_ptr());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f;
    input_data[3] = 4.0f; input_data[4] = 5.0f; input_data[5] = 6.0f;
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    print_matrix("Input", input);
    
    Tensor output = layer.forward(input);
    print_matrix("Output", output);
    
    Tensor grad_output({2, 2}, DType::Float32, DeviceType::CPU);
    float* grad_out_data = static_cast<float*>(grad_output.data_ptr());
    grad_out_data[0] = 1.0f; grad_out_data[1] = 1.0f;
    grad_out_data[2] = 1.0f; grad_out_data[3] = 1.0f;
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    print_matrix("Grad Output", grad_output);
    
    Tensor grad_input = layer.backward(grad_output);
    
    std::cout << "\n--- Gradients ---" << std::endl;
    print_matrix("Grad Input", grad_input);
    print_matrix("Grad Weight", layer.grad_weight());
    print_tensor("Grad Bias", layer.grad_bias());
    
    std::cout << "\n--- Expected Calculations ---" << std::endl;
    std::cout << "Grad Input = Grad Output @ Weight^T" << std::endl;
    std::cout << "  All grad_input values should be 1.0" << std::endl;
    std::cout << "\nGrad Weight = Input^T @ Grad Output" << std::endl;
    std::cout << "  grad_weight = [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]" << std::endl;
    std::cout << "\nGrad Bias = sum(Grad Output, dim=0)" << std::endl;
    std::cout << "  grad_bias = [2.0, 2.0]" << std::endl;
    
    bool shape_ok = (grad_input.shape() == std::vector<int64_t>{2, 3});
    
    float* grad_input_data = static_cast<float*>(grad_input.data_ptr());
    bool grad_input_ok = true;
    for (int i = 0; i < 6; ++i) {
        if (std::abs(grad_input_data[i] - 1.0f) > 1e-5) {
            grad_input_ok = false;
            std::cout << "  Grad input mismatch at " << i << ": " 
                      << grad_input_data[i] << " vs 1.0" << std::endl;
            break;
        }
    }
    
    Tensor grad_weight_cpu = layer.grad_weight().to(DeviceType::CPU);
    float* grad_weight_data = static_cast<float*>(grad_weight_cpu.data_ptr());
    float expected_grad_weight[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
    bool grad_weight_ok = true;
    for (int i = 0; i < 6; ++i) {
        if (std::abs(grad_weight_data[i] - expected_grad_weight[i]) > 1e-5) {
            grad_weight_ok = false;
            std::cout << "  Grad weight mismatch at " << i << ": " 
                      << grad_weight_data[i] << " vs " << expected_grad_weight[i] << std::endl;
            break;
        }
    }
    
    Tensor grad_bias_cpu = layer.grad_bias().to(DeviceType::CPU);
    float* grad_bias_data = static_cast<float*>(grad_bias_cpu.data_ptr());
    bool grad_bias_ok = (std::abs(grad_bias_data[0] - 2.0f) < 1e-5 &&
                         std::abs(grad_bias_data[1] - 2.0f) < 1e-5);
    
    if (!grad_bias_ok) {
        std::cout << "  Grad bias mismatch: [" << grad_bias_data[0] << ", " 
                  << grad_bias_data[1] << "] vs [2.0, 2.0]" << std::endl;
    }
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Backward CPU - Shape", shape_ok);
    print_test_result("Backward CPU - Grad Input", grad_input_ok);
    print_test_result("Backward CPU - Grad Weight", grad_weight_ok);
    print_test_result("Backward CPU - Grad Bias", grad_bias_ok);
}

// 测试3：前向传播 (CUDA)
void test_forward_cuda_detailed() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 3: Forward Pass (CUDA) - Detailed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer(3, 2);
    init_weights(layer.weight(), layer.bias(), 0.5f, 0.1f);
    
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    print_matrix("Weight", layer.weight());
    print_tensor("Bias", layer.bias());
    
    Tensor input({2, 3}, DType::Float32, Device(DeviceType::CUDA, 0));
    Tensor input_cpu({2, 3}, DType::Float32, DeviceType::CPU);
    float* input_data = static_cast<float*>(input_cpu.data_ptr());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f;
    input_data[3] = 4.0f; input_data[4] = 5.0f; input_data[5] = 6.0f;
    input = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    std::cout << "\n--- Input ---" << std::endl;
    print_matrix("Input", input_cpu);
    
    Tensor output = layer.forward(input);
    Tensor output_cpu = output.to(DeviceType::CPU);
    
    std::cout << "\n--- Output ---" << std::endl;
    print_matrix("Output", output_cpu);
    
    bool shape_ok = (output.shape() == std::vector<int64_t>{2, 2});
    float* output_data = static_cast<float*>(output_cpu.data_ptr());
    bool values_ok = (std::abs(output_data[0] - 3.1f) < 1e-4 &&
                      std::abs(output_data[1] - 3.1f) < 1e-4 &&
                      std::abs(output_data[2] - 7.6f) < 1e-4 &&
                      std::abs(output_data[3] - 7.6f) < 1e-4);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Forward CUDA - Shape", shape_ok);
    print_test_result("Forward CUDA - Values", values_ok);
}

// 测试4：反向传播 (CUDA)
void test_backward_cuda_detailed() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 4: Backward Pass (CUDA) - Detailed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer(3, 2);
    init_weights(layer.weight(), layer.bias(), 0.5f, 0.1f);
    
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    print_matrix("Weight", layer.weight());
    print_tensor("Bias", layer.bias());
    
    Tensor input({2, 3}, DType::Float32, Device(DeviceType::CUDA, 0));
    Tensor input_cpu({2, 3}, DType::Float32, DeviceType::CPU);
    float* input_data = static_cast<float*>(input_cpu.data_ptr());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f;
    input_data[3] = 4.0f; input_data[4] = 5.0f; input_data[5] = 6.0f;
    input = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    print_matrix("Input", input_cpu);
    
    Tensor output = layer.forward(input);
    print_matrix("Output", output);
    
    Tensor grad_output({2, 2}, DType::Float32, Device(DeviceType::CUDA, 0));
    Tensor grad_cpu({2, 2}, DType::Float32, DeviceType::CPU);
    float* grad_data = static_cast<float*>(grad_cpu.data_ptr());
    grad_data[0] = 1.0f; grad_data[1] = 1.0f;
    grad_data[2] = 1.0f; grad_data[3] = 1.0f;
    grad_output = grad_cpu.to(Device(DeviceType::CUDA, 0));
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    print_matrix("Grad Output", grad_cpu);
    
    Tensor grad_input = layer.backward(grad_output);
    
    std::cout << "\n--- Gradients ---" << std::endl;
    print_matrix("Grad Input", grad_input);
    print_matrix("Grad Weight", layer.grad_weight());
    print_tensor("Grad Bias", layer.grad_bias());
    
    Tensor grad_input_cpu = grad_input.to(DeviceType::CPU);
    bool shape_ok = (grad_input.shape() == std::vector<int64_t>{2, 3});
    
    float* grad_input_data = static_cast<float*>(grad_input_cpu.data_ptr());
    bool grad_input_ok = true;
    for (int i = 0; i < 6; ++i) {
        if (std::abs(grad_input_data[i] - 1.0f) > 1e-4) {
            grad_input_ok = false;
            std::cout << "  Grad input mismatch at " << i << ": " 
                      << grad_input_data[i] << " vs 1.0" << std::endl;
            break;
        }
    }
    
    Tensor grad_weight_cpu = layer.grad_weight().to(DeviceType::CPU);
    float* grad_weight_data = static_cast<float*>(grad_weight_cpu.data_ptr());
    float expected_grad_weight[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
    bool grad_weight_ok = true;
    for (int i = 0; i < 6; ++i) {
        if (std::abs(grad_weight_data[i] - expected_grad_weight[i]) > 1e-4) {
            grad_weight_ok = false;
            std::cout << "  Grad weight mismatch at " << i << ": " 
                      << grad_weight_data[i] << " vs " << expected_grad_weight[i] << std::endl;
            break;
        }
    }
    
    Tensor grad_bias_cpu = layer.grad_bias().to(DeviceType::CPU);
    float* grad_bias_data = static_cast<float*>(grad_bias_cpu.data_ptr());
    bool grad_bias_ok = (std::abs(grad_bias_data[0] - 2.0f) < 1e-4 &&
                         std::abs(grad_bias_data[1] - 2.0f) < 1e-4);
    
    if (!grad_bias_ok) {
        std::cout << "  Grad bias mismatch: [" << grad_bias_data[0] << ", " 
                  << grad_bias_data[1] << "] vs [2.0, 2.0]" << std::endl;
    }
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Backward CUDA - Shape", shape_ok);
    print_test_result("Backward CUDA - Grad Input", grad_input_ok);
    print_test_result("Backward CUDA - Grad Weight", grad_weight_ok);
    print_test_result("Backward CUDA - Grad Bias", grad_bias_ok);
}

// 测试5：CPU vs CUDA 一致性
void test_cpu_cuda_consistency() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 5: CPU vs CUDA Consistency" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer_cpu(3, 2);
    Linear layer_cuda(3, 2);
    
    init_weights(layer_cpu.weight(), layer_cpu.bias(), 0.3f, 0.05f);
    init_weights(layer_cuda.weight(), layer_cuda.bias(), 0.3f, 0.05f);
    
    Tensor input_cpu({4, 3}, DType::Float32, DeviceType::CPU);
    float* data = static_cast<float*>(input_cpu.data_ptr());
    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        data[i] = static_cast<float>(i) * 0.1f;
    }
    
    std::cout << "\n--- Input ---" << std::endl;
    print_matrix("Input", input_cpu);
    
    Tensor input_cuda = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    Tensor output_cpu = layer_cpu.forward(input_cpu);
    Tensor output_cuda = layer_cuda.forward(input_cuda);
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    print_matrix("CPU Output", output_cpu);
    print_matrix("CUDA Output", output_cuda);
    
    bool forward_match = tensors_close(output_cpu, output_cuda, 1e-4, 1e-6);
    
    Tensor grad_output_cpu({4, 2}, DType::Float32, DeviceType::CPU);
    float* grad_data = static_cast<float*>(grad_output_cpu.data_ptr());
    for (int64_t i = 0; i < grad_output_cpu.numel(); ++i) {
        grad_data[i] = 1.0f;
    }
    Tensor grad_output_cuda = grad_output_cpu.to(Device(DeviceType::CUDA, 0));
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    print_matrix("Grad Output", grad_output_cpu);
    
    Tensor grad_input_cpu = layer_cpu.backward(grad_output_cpu);
    Tensor grad_input_cuda = layer_cuda.backward(grad_output_cuda);
    
    print_matrix("CPU Grad Input", grad_input_cpu);
    print_matrix("CUDA Grad Input", grad_input_cuda);
    
    std::cout << "\n--- Weight Gradients ---" << std::endl;
    print_matrix("CPU Grad Weight", layer_cpu.grad_weight());
    print_matrix("CUDA Grad Weight", layer_cuda.grad_weight());
    
    std::cout << "\n--- Bias Gradients ---" << std::endl;
    print_tensor("CPU Grad Bias", layer_cpu.grad_bias());
    print_tensor("CUDA Grad Bias", layer_cuda.grad_bias());
    
    bool backward_match = tensors_close(grad_input_cpu, grad_input_cuda, 1e-4, 1e-6);
    bool grad_weight_match = tensors_close(layer_cpu.grad_weight(), layer_cuda.grad_weight(), 1e-4, 1e-6);
    bool grad_bias_match = tensors_close(layer_cpu.grad_bias(), layer_cuda.grad_bias(), 1e-4, 1e-6);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("CPU vs CUDA - Forward", forward_match);
    print_test_result("CPU vs CUDA - Backward Input", backward_match);
    print_test_result("CPU vs CUDA - Grad Weight", grad_weight_match);
    print_test_result("CPU vs CUDA - Grad Bias", grad_bias_match);
}

// 测试6：不同维度
void test_different_dimensions() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 6: Different Dimensions (5x7 -> 7x3)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    int64_t batch = 5, in_features = 7, out_features = 3;
    
    Linear layer_cpu(in_features, out_features);
    Linear layer_cuda(in_features, out_features);
    
    init_weights(layer_cpu.weight(), layer_cpu.bias(), 0.2f, 0.05f);
    init_weights(layer_cuda.weight(), layer_cuda.bias(), 0.2f, 0.05f);
    
    Tensor input_cpu({batch, in_features}, DType::Float32, DeviceType::CPU);
    float* data = static_cast<float*>(input_cpu.data_ptr());
    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        data[i] = static_cast<float>(i) * 0.05f - 0.5f;
    }
    
    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Batch: " << batch << ", In: " << in_features << ", Out: " << out_features << std::endl;
    print_matrix("Input", input_cpu);
    
    Tensor input_cuda = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    Tensor output_cpu = layer_cpu.forward(input_cpu);
    Tensor output_cuda = layer_cuda.forward(input_cuda);
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    print_matrix("CPU Output", output_cpu);
    print_matrix("CUDA Output", output_cuda);
    
    bool forward_match = tensors_close(output_cpu, output_cuda, 1e-4, 1e-6);
    
    Tensor grad_output_cpu({batch, out_features}, DType::Float32, DeviceType::CPU);
    float* grad_data = static_cast<float*>(grad_output_cpu.data_ptr());
    for (int64_t i = 0; i < grad_output_cpu.numel(); ++i) {
        grad_data[i] = 1.0f;
    }
    Tensor grad_output_cuda = grad_output_cpu.to(Device(DeviceType::CUDA, 0));
    
    Tensor grad_input_cpu = layer_cpu.backward(grad_output_cpu);
    Tensor grad_input_cuda = layer_cuda.backward(grad_output_cuda);
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    print_matrix("CPU Grad Input", grad_input_cpu);
    print_matrix("CUDA Grad Input", grad_input_cuda);
    
    std::cout << "\n--- Gradients Comparison ---" << std::endl;
    print_matrix("CPU Grad Weight", layer_cpu.grad_weight());
    print_matrix("CUDA Grad Weight", layer_cuda.grad_weight());
    print_tensor("CPU Grad Bias", layer_cpu.grad_bias());
    print_tensor("CUDA Grad Bias", layer_cuda.grad_bias());
    
    bool backward_match = tensors_close(grad_input_cpu, grad_input_cuda, 1e-4, 1e-6);
    bool grad_weight_match = tensors_close(layer_cpu.grad_weight(), layer_cuda.grad_weight(), 1e-4, 1e-6);
    bool grad_bias_match = tensors_close(layer_cpu.grad_bias(), layer_cuda.grad_bias(), 1e-4, 1e-6);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Different Dims - Forward", forward_match);
    print_test_result("Different Dims - Backward Input", backward_match);
    print_test_result("Different Dims - Grad Weight", grad_weight_match);
    print_test_result("Different Dims - Grad Bias", grad_bias_match);
}

// 测试7：大批量
void test_large_batch() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 7: Large Batch (128x64 -> 64x32)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    int64_t batch = 128, in_features = 64, out_features = 32;
    
    Linear layer_cpu(in_features, out_features);
    Linear layer_cuda(in_features, out_features);
    
    init_weights(layer_cpu.weight(), layer_cpu.bias(), 0.1f, 0.01f);
    init_weights(layer_cuda.weight(), layer_cuda.bias(), 0.1f, 0.01f);
    
    Tensor input_cpu({batch, in_features}, DType::Float32, DeviceType::CPU);
    float* data = static_cast<float*>(input_cpu.data_ptr());
    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        data[i] = static_cast<float>(i % 100) * 0.01f;
    }
    
    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Batch: " << batch << ", In: " << in_features << ", Out: " << out_features << std::endl;
    print_matrix("Input (first 5 rows)", input_cpu, 5, 8);
    
    Tensor input_cuda = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    // 计时
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor output_cpu = layer_cpu.forward(input_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    auto start_cuda = std::chrono::high_resolution_clock::now();
    Tensor output_cuda = layer_cuda.forward(input_cuda);
    cudaDeviceSynchronize();
    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda);
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    std::cout << "CPU Time: " << duration_cpu.count() << " μs" << std::endl;
    std::cout << "CUDA Time: " << duration_cuda.count() << " μs" << std::endl;
    std::cout << "Speedup: " << (float)duration_cpu.count() / duration_cuda.count() << "x" << std::endl;
    
    print_matrix("CPU Output (first 3 rows)", output_cpu, 3, 8);
    print_matrix("CUDA Output (first 3 rows)", output_cuda, 3, 8);
    
    bool forward_match = tensors_close(output_cpu, output_cuda, 1e-4, 1e-6);
    
    Tensor grad_output_cpu({batch, out_features}, DType::Float32, DeviceType::CPU);
    float* grad_data = static_cast<float*>(grad_output_cpu.data_ptr());
    for (int64_t i = 0; i < grad_output_cpu.numel(); ++i) {
        grad_data[i] = 0.5f;
    }
    Tensor grad_output_cuda = grad_output_cpu.to(Device(DeviceType::CUDA, 0));
    
    start_cpu = std::chrono::high_resolution_clock::now();
    Tensor grad_input_cpu = layer_cpu.backward(grad_output_cpu);
    end_cpu = std::chrono::high_resolution_clock::now();
    duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    start_cuda = std::chrono::high_resolution_clock::now();
    Tensor grad_input_cuda = layer_cuda.backward(grad_output_cuda);
    cudaDeviceSynchronize();
    end_cuda = std::chrono::high_resolution_clock::now();
    duration_cuda = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda);
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    std::cout << "CPU Time: " << duration_cpu.count() << " μs" << std::endl;
    std::cout << "CUDA Time: " << duration_cuda.count() << " μs" << std::endl;
    std::cout << "Speedup: " << (float)duration_cpu.count() / duration_cuda.count() << "x" << std::endl;
    
    print_matrix("CPU Grad Weight (first 5 rows)", layer_cpu.grad_weight(), 5, 8);
    print_matrix("CUDA Grad Weight (first 5 rows)", layer_cuda.grad_weight(), 5, 8);
    print_tensor("CPU Grad Bias (first 10)", layer_cpu.grad_bias(), 10);
    print_tensor("CUDA Grad Bias (first 10)", layer_cuda.grad_bias(), 10);
    
    bool backward_match = tensors_close(grad_input_cpu, grad_input_cuda, 1e-4, 1e-6);
    bool grad_weight_match = tensors_close(layer_cpu.grad_weight(), layer_cuda.grad_weight(), 1e-4, 1e-6);
    bool grad_bias_match = tensors_close(layer_cpu.grad_bias(), layer_cuda.grad_bias(), 1e-4, 1e-6);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Large Batch - Forward", forward_match);
    print_test_result("Large Batch - Backward Input", backward_match);
    print_test_result("Large Batch - Grad Weight", grad_weight_match);
    print_test_result("Large Batch - Grad Bias", grad_bias_match);
}

// 测试8：负值
void test_negative_values() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 8: Negative Values (4x3 -> 3x2)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer_cpu(3, 2);
    Linear layer_cuda(3, 2);
    
    init_weights(layer_cpu.weight(), layer_cpu.bias(), 0.4f, -0.1f);
    init_weights(layer_cuda.weight(), layer_cuda.bias(), 0.4f, -0.1f);
    
    Tensor input_cpu({4, 3}, DType::Float32, DeviceType::CPU);
    float* data = static_cast<float*>(input_cpu.data_ptr());
    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        data[i] = static_cast<float>(i) * 0.3f - 2.0f;
    }
    
    std::cout << "\n--- Input ---" << std::endl;
    print_matrix("Input", input_cpu);
    
    Tensor input_cuda = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    Tensor output_cpu = layer_cpu.forward(input_cpu);
    Tensor output_cuda = layer_cuda.forward(input_cuda);
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    print_matrix("CPU Output", output_cpu);
    print_matrix("CUDA Output", output_cuda);
    
    bool forward_match = tensors_close(output_cpu, output_cuda, 1e-4, 1e-6);
    
    Tensor grad_output_cpu({4, 2}, DType::Float32, DeviceType::CPU);
    float* grad_data = static_cast<float*>(grad_output_cpu.data_ptr());
    for (int64_t i = 0; i < grad_output_cpu.numel(); ++i) {
        grad_data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    Tensor grad_output_cuda = grad_output_cpu.to(Device(DeviceType::CUDA, 0));
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    print_matrix("Grad Output", grad_output_cpu);
    
    Tensor grad_input_cpu = layer_cpu.backward(grad_output_cpu);
    Tensor grad_input_cuda = layer_cuda.backward(grad_output_cuda);
    
    print_matrix("CPU Grad Input", grad_input_cpu);
    print_matrix("CUDA Grad Input", grad_input_cuda);
    
    std::cout << "\n--- Gradients Comparison ---" << std::endl;
    print_matrix("CPU Grad Weight", layer_cpu.grad_weight());
    print_matrix("CUDA Grad Weight", layer_cuda.grad_weight());
    print_tensor("CPU Grad Bias", layer_cpu.grad_bias());
    print_tensor("CUDA Grad Bias", layer_cuda.grad_bias());
    
    bool backward_match = tensors_close(grad_input_cpu, grad_input_cuda, 1e-4, 1e-6);
    bool grad_weight_match = tensors_close(layer_cpu.grad_weight(), layer_cuda.grad_weight(), 1e-4, 1e-6);
    bool grad_bias_match = tensors_close(layer_cpu.grad_bias(), layer_cuda.grad_bias(), 1e-4, 1e-6);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Negative Values - Forward", forward_match);
    print_test_result("Negative Values - Backward Input", backward_match);
    print_test_result("Negative Values - Grad Weight", grad_weight_match);
    print_test_result("Negative Values - Grad Bias", grad_bias_match);
}

// 测试9：单样本
void test_single_sample() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 9: Single Sample (1x10 -> 10x5)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    Linear layer_cpu(10, 5);
    Linear layer_cuda(10, 5);
    
    init_weights(layer_cpu.weight(), layer_cpu.bias(), 0.15f, 0.05f);
    init_weights(layer_cuda.weight(), layer_cuda.bias(), 0.15f, 0.05f);
    
    Tensor input_cpu({1, 10}, DType::Float32, DeviceType::CPU);
    float* data = static_cast<float*>(input_cpu.data_ptr());
    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        data[i] = static_cast<float>(i + 1) * 0.1f;
    }
    
    std::cout << "\n--- Input ---" << std::endl;
    print_tensor("Input", input_cpu);
    
    Tensor input_cuda = input_cpu.to(Device(DeviceType::CUDA, 0));
    
    Tensor output_cpu = layer_cpu.forward(input_cpu);
    Tensor output_cuda = layer_cuda.forward(input_cuda);
    
    std::cout << "\n--- Forward Pass ---" << std::endl;
    print_tensor("CPU Output", output_cpu);
    print_tensor("CUDA Output", output_cuda);
    
    bool forward_match = tensors_close(output_cpu, output_cuda, 1e-4, 1e-6);
    
    Tensor grad_output_cpu({1, 5}, DType::Float32, DeviceType::CPU);
    float* grad_data = static_cast<float*>(grad_output_cpu.data_ptr());
    for (int64_t i = 0; i < grad_output_cpu.numel(); ++i) {
        grad_data[i] = 2.0f;
    }
    Tensor grad_output_cuda = grad_output_cpu.to(Device(DeviceType::CUDA, 0));
    
    Tensor grad_input_cpu = layer_cpu.backward(grad_output_cpu);
    Tensor grad_input_cuda = layer_cuda.backward(grad_output_cuda);
    
    std::cout << "\n--- Backward Pass ---" << std::endl;
    print_tensor("CPU Grad Input", grad_input_cpu);
    print_tensor("CUDA Grad Input", grad_input_cuda);
    
    std::cout << "\n--- Gradients Comparison ---" << std::endl;
    print_matrix("CPU Grad Weight", layer_cpu.grad_weight());
    print_matrix("CUDA Grad Weight", layer_cuda.grad_weight());
    print_tensor("CPU Grad Bias", layer_cpu.grad_bias());
    print_tensor("CUDA Grad Bias", layer_cuda.grad_bias());
    
    bool backward_match = tensors_close(grad_input_cpu, grad_input_cuda, 1e-4, 1e-6);
    bool grad_weight_match = tensors_close(layer_cpu.grad_weight(), layer_cuda.grad_weight(), 1e-4, 1e-6);
    bool grad_bias_match = tensors_close(layer_cpu.grad_bias(), layer_cuda.grad_bias(), 1e-4, 1e-6);
    
    std::cout << "\n--- Test Results ---" << std::endl;
    print_test_result("Single Sample - Forward", forward_match);
    print_test_result("Single Sample - Backward Input", backward_match);
    print_test_result("Single Sample - Grad Weight", grad_weight_match);
    print_test_result("Single Sample - Grad Bias", grad_bias_match);
}

// ============ Main ============

int main() {
    std::cout << "\n";
    std::cout << "########################################################" << std::endl;
    std::cout << "#                                                      #" << std::endl;
    std::cout << "#         Linear Layer Test Suite (Extended)          #" << std::endl;
    std::cout << "#                                                      #" << std::endl;
    std::cout << "########################################################" << std::endl;
    
    try {
        // 基础测试
        test_forward_cpu_detailed();
        test_backward_cpu_detailed();
        test_forward_cuda_detailed();
        test_backward_cuda_detailed();
        
        // 一致性测试
        test_cpu_cuda_consistency();
        
        // 扩展测试
        test_different_dimensions();
        test_large_batch();
        test_negative_values();
        test_single_sample();
        
        std::cout << "\n";
        std::cout << "########################################################" << std::endl;
        std::cout << "#                                                      #" << std::endl;
        std::cout << "#              All Tests Completed!                    #" << std::endl;
        std::cout << "#                                                      #" << std::endl;
        std::cout << "########################################################" << std::endl;
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception caught: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
