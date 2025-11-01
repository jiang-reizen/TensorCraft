#pragma once
#include <cstddef>
#include <string>

namespace framework {

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
    Bool
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::UInt8: return 1;
        case DType::Bool: return 1;
        default: return 0;
    }
}

inline std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt8: return "uint8";
        case DType::Bool: return "bool";
        default: return "unknown";
    }
}

} // namespace framework
