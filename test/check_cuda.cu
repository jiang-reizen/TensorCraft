#include <iostream>

int main() {
#ifdef CUDA_AVAILABLE
    std::cout << "CUDA_AVAILABLE is defined!" << std::endl;
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess) {
        std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        }
    } else {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
#else
    std::cout << "CUDA_AVAILABLE is NOT defined!" << std::endl;
#endif
    
    return 0;
}
