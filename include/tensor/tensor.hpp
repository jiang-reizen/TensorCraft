#pragma once
#include "storage.hpp"
#include <vector>
#include <initializer_list>
#include <memory>
#include <functional>

namespace framework {

class Tensor {
public:
    // 构造函数
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::Float32, 
           const Device& device = Device(DeviceType::CPU));
    
    // 从现有 storage 创建（用于视图操作）
    Tensor(std::shared_ptr<Storage> storage, 
           const std::vector<int64_t>& shape,
           const std::vector<int64_t>& stride,
           int64_t offset = 0);

    // 基本属性
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& stride() const { return stride_; }
    int64_t offset() const { return offset_; }
    int64_t ndim() const { return shape_.size(); }
    int64_t numel() const;
    DType dtype() const { return storage_->dtype(); }
    const Device& device() const { return storage_->device(); }
    bool is_contiguous() const;

    // 数据访问
    void* data_ptr() { return static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype()); }
    const void* data_ptr() const { return static_cast<const char*>(storage_->data()) + offset_ * dtype_size(dtype()); }

    // 内存管理
    Tensor clone() const;  // 深拷贝
    Tensor contiguous() const;  // 返回连续内存的tensor
    Tensor to(const Device& device) const;  // 设备转换
    Tensor to(DType dtype) const;  // 类型转换

    // 视图操作（浅拷贝，共享storage）
    Tensor view(const std::vector<int64_t>& shape) const;
    Tensor reshape(const std::vector<int64_t>& shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor squeeze(int64_t dim = -1) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    
    // 索引操作
    Tensor operator[](int64_t index) const;
    Tensor index(const std::vector<int64_t>& indices) const;

    // 数据填充
    void fill_(float value);
    void zero_();
    void ones_();
    
    // 工厂方法
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32,
                       const Device& device = Device(DeviceType::CPU));
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32,
                      const Device& device = Device(DeviceType::CPU));
    static Tensor randn(const std::vector<int64_t>& shape, DType dtype = DType::Float32,
                       const Device& device = Device(DeviceType::CPU));
    static Tensor arange(float start, float end, float step = 1.0f, DType dtype = DType::Float32,
                        const Device& device = Device(DeviceType::CPU));
    static Tensor from_blob(void* data, const std::vector<int64_t>& shape, DType dtype,
                           const Device& device = Device(DeviceType::CPU));

    // 基本运算（返回新tensor）
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // 标量运算
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // inplace 运算
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // 矩阵运算
    Tensor matmul(const Tensor& other) const;
    Tensor mm(const Tensor& other) const;  // 2D 矩阵乘法

    // 归约操作
    Tensor sum(int64_t dim = -1, bool keepdim = false) const;
    Tensor mean(int64_t dim = -1, bool keepdim = false) const;
    Tensor max(int64_t dim = -1, bool keepdim = false) const;
    Tensor min(int64_t dim = -1, bool keepdim = false) const;

    // 激活函数
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int64_t dim = -1) const;

    // 用于后续扩展自动微分的接口
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    // 预留梯度指针（后续实现自动微分时使用）
    std::shared_ptr<Tensor> grad() const { return grad_; }
    void set_grad(std::shared_ptr<Tensor> grad) { grad_ = grad; }
    
    // 预留反向传播函数（后续实现自动微分时使用）
    using BackwardFunction = std::function<void()>;
    void set_grad_fn(BackwardFunction fn) { grad_fn_ = fn; }
    BackwardFunction grad_fn() const { return grad_fn_; }

    // 调试
    void print() const;
    std::string str() const;

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;
    int64_t offset_;

    // 自动微分相关（预留）
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    BackwardFunction grad_fn_;

    // 辅助函数
    static std::vector<int64_t> compute_stride(const std::vector<int64_t>& shape);
    int64_t compute_offset(const std::vector<int64_t>& indices) const;
    void check_device_match(const Tensor& other) const;
    void check_shape_match(const Tensor& other) const;
};

} // namespace framework
