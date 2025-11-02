#include "layers/linear.hpp"
#include <vector>
#include <cassert>

namespace Layers {

Linear::Linear(int64_t in_features, int64_t out_features) :
    c_in(in_features), c_out(out_features),
    weight_({in_features, out_features}),
    bias_({out_features}),
    grad_weight_({in_features, out_features}),
    grad_bias_({out_features}) {
    // how to initialize weight and bias?
}

void matrix_multiply_cpu(
    int M, int N, int K,
    const float* srcA, const float* srcB, float* dst) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += srcA[i * K + k] * srcB[k * N + j];
            dst[i * N + j] = sum;
        }
}

void add_bias_cpu(
    int M, int N,
    const float* src,
    float* dst) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            dst[i * N + j] += src[j];
}

void accelerate_bias_cpu(
    int M, int N,
    const float* src,
    float* dst) {
    for (int j = 0; j < N; ++j)
        dst[j] = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            dst[j] += src[i * N + j];
}

__global__ void matrix_multiply_cuda(
    int M, int N, int K,
    const float* srcA, const float* srcB, float* dst) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k)
            sum += srcA[row * K + k] * srcB[k * N + col];
        dst[row * N + col] = sum;
    }
}

__global__ void add_bias_cuda(
    int M, int N,
    const float* src,
    float* dst) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        dst[row * N + col] += src[col];
    }
}

// ✅ 修复：使用 atomicAdd 避免竞态条件
__global__ void accelerate_bias_cuda(
    int M, int N,
    const float* src,
    float* dst) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        atomicAdd(&dst[col], src[row * N + col]);
    }
}

Tensor Linear::forward(const Tensor &input) {
    // assume input = (batch, c_in), dtype == Float32, contiguous
    assert(input.shape().size() == 2);
    assert(input.dtype() == DType::Float32);
    assert(input.is_contiguous());

    para.emplace_back(input.clone());

    weight_ = weight_.to(input.device());
    bias_ = bias_.to(input.device());
    int64_t batch = input.shape()[0];
    Tensor output({batch, c_out}, DType::Float32, input.device());
    float* in_data = (float*)input.data_ptr();
    float* weight_data = (float*)weight_.data_ptr();
    float* bias_data = (float*)bias_.data_ptr();
    float* out_data = (float*)output.data_ptr();
    
    if (input.device().is_cpu()) {
        matrix_multiply_cpu(batch, c_out, c_in, in_data, weight_data, out_data);
        add_bias_cpu(batch, c_out, bias_data, out_data);
        return output;
    }
    else {
        cudaSetDevice(input.device().index());

        dim3 gridDim((c_out + 31) / 32, (batch + 31) / 32), blockDim(32, 32);
        matrix_multiply_cuda<<<gridDim, blockDim>>>(batch, c_out, c_in, in_data, weight_data, out_data);
        cudaDeviceSynchronize();
        add_bias_cuda<<<gridDim, blockDim>>>(batch, c_out, bias_data, out_data);
        cudaDeviceSynchronize();
    }

    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // assume output = (batch, c_out), dtype == Float32, contiguous
    assert(grad_output.shape().size() == 2);
    assert(grad_output.dtype() == DType::Float32);
    assert(grad_output.is_contiguous());

    // ✅ 确保所有张量在同一设备
    grad_weight_ = grad_weight_.to(grad_output.device());
    grad_bias_ = grad_bias_.to(grad_output.device());
    weight_ = weight_.to(grad_output.device());

    int64_t batch = grad_output.shape()[0];
    Tensor grad_input({batch, c_in}, DType::Float32, grad_output.device());
    Tensor x_trans = para.back(); 
    para.pop_back(); 
    x_trans = x_trans.to(grad_output.device());
    x_trans = x_trans.transpose(0, 1);
    x_trans = x_trans.contiguous();

    // ✅ 修复：创建正确的转置权重
    Tensor w_trans = weight_.transpose(0, 1);
    w_trans = w_trans.contiguous();  // 确保连续

    float* grad_output_data = (float*)grad_output.data_ptr();
    float* grad_input_data = (float*)grad_input.data_ptr();
    float* weight_trans_data = (float*)w_trans.data_ptr();  // ✅ 修复
    float* grad_bias_data = (float*)grad_bias_.data_ptr();
    float* grad_weight_data = (float*)grad_weight_.data_ptr();
    float* input_trans_data = (float*)x_trans.data_ptr();

    if (grad_output.device().is_cpu()) {
        matrix_multiply_cpu(batch, c_in, c_out, grad_output_data, weight_trans_data, grad_input_data);
        matrix_multiply_cpu(c_in, c_out, batch, input_trans_data, grad_output_data, grad_weight_data);
        accelerate_bias_cpu(batch, c_out, grad_output_data, grad_bias_data);
    }
    else {
        cudaSetDevice(grad_output.device().index());
        
        // grad_input = grad_output @ w_trans
        dim3 gridDim1((c_in + 31) / 32, (batch + 31) / 32), blockDim(32, 32);
        matrix_multiply_cuda<<<gridDim1, blockDim>>>(batch, c_in, c_out, grad_output_data, weight_trans_data, grad_input_data);
        
        // grad_weight = x_trans @ grad_output
        dim3 gridDim2((c_out + 31) / 32, (c_in + 31) / 32);
        matrix_multiply_cuda<<<gridDim2, blockDim>>>(c_in, c_out, batch, input_trans_data, grad_output_data, grad_weight_data);
        
        // grad_bias = sum(grad_output, dim=0)
        // ✅ 修复：先清零，再累加
        cudaMemset(grad_bias_data, 0, c_out * sizeof(float));  // ✅ 修复
        dim3 gridDim3((c_out + 31) / 32, (batch + 31) / 32);
        accelerate_bias_cuda<<<gridDim3, blockDim>>>(batch, c_out, grad_output_data, grad_bias_data);
        
        cudaDeviceSynchronize();
    }

    return grad_input;
}

} // namespace Layers
