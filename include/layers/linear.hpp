#pragma once

#include "layers/layers.hpp"
using namespace TensorFramework;

namespace Layers {

class Linear : public Layer {
private:
    int64_t c_in, c_out;
    Tensor weight_; // shape (c_out, c_in)
    Tensor bias_; // shape (c_out)

    Tensor grad_weight_;
    Tensor grad_bias_;
    std::vector<Tensor> para;
public:
    Linear(int64_t in_features, int64_t out_features);

    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    Tensor& grad_weight() { return grad_weight_; }
    Tensor& grad_bias() { return grad_bias_; }
};

}