#pragma once

#include "tensor/tensor.hpp"
#include <vector>
#include <memory>

using namespace TensorFramework;

namespace Layers {

class Layer {
public:
    virtual ~Layer() = default;
    
    // Forward pass
    virtual Tensor forward(const Tensor& input) = 0;
    
    // Backward pass
    virtual Tensor backward(const Tensor& grad_output) = 0;
    

    // Get parameters
    // virtual std::vector<Tensor*> parameters() { return {}; }
    
    // Get gradients
    // virtual std::vector<Tensor*> gradients() { return {}; }
    
    // Training mode
    // virtual void train() { training_ = true; }
    // virtual void eval() { training_ = false; }
    // bool is_training() const { return training_; }
    
protected:
    // bool training_ = true;
};

} // namespace Layers
