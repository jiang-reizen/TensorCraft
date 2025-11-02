-- 设置项目
set_project("TensorCraft")
set_version("0.1.0")

-- 设置C++标准
set_languages("c++17")

-- 添加编译模式
add_rules("mode.debug", "mode.release")

-- Windows特定配置
if is_plat("windows") then
    add_cxxflags("/utf-8")
    add_cuflags("-Xcompiler=/utf-8")
    add_defines("NOMINMAX", "_CRT_SECURE_NO_WARNINGS")
end

-- 定义tensor库
target("tensor")
    set_kind("static")
    
    -- 添加源文件
    add_files("src/tensor/*.cu")
    
    -- 添加头文件搜索路径
    add_includedirs("include", {public = true})
    
    -- CUDA相关配置
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86", "compute_89")
    
    -- CUDA编译选项
    add_cuflags("-use_fast_math")
    add_cuflags("--expt-relaxed-constexpr")
    
    -- 链接CUDA运行时
    add_links("cudart")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end
    
    -- 定义宏
    add_defines("CUDA_AVAILABLE")
    if is_mode("debug") then
        add_defines("DEBUG")
    end

-- 定义layers库
target("layers")
    set_kind("static")
    
    -- 添加源文件
    add_files("src/layers/*.cu")
    
    -- 添加头文件搜索路径
    add_includedirs("include", {public = true})
    
    -- 依赖tensor库
    add_deps("tensor")
    
    -- CUDA相关配置
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86", "compute_89")
    
    -- CUDA编译选项
    add_cuflags("-use_fast_math")
    add_cuflags("--expt-relaxed-constexpr")
    
    -- 链接CUDA运行时和cuBLAS
    add_links("cudart", "cublas")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end
    
    add_defines("CUDA_AVAILABLE")

-- 定义测试目标：test_tensor
target("test_tensor")
    set_kind("binary")
    set_default(false)
    
    -- 添加测试源文件
    add_files("test/test_tensor.cu")
    
    -- 依赖tensor库
    add_deps("tensor")
    
    -- CUDA相关配置
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86", "compute_89")
    
    -- 添加头文件搜索路径
    add_includedirs("include")
    
    -- 链接CUDA运行时
    add_links("cudart")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end

    add_defines("CUDA_AVAILABLE")
    
    after_build(function (target)
        print("[Ok] test_tensor built successfully!")
        print("Run with: xmake run test_tensor")
    end)

-- 定义测试目标：test_linear
target("test_linear")
    set_kind("binary")
    set_default(false)
    
    -- 添加测试源文件
    add_files("test/test_linear.cu")
    
    -- 依赖tensor和layers库
    add_deps("tensor", "layers")
    
    -- CUDA相关配置
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86", "compute_89")
    
    -- 添加头文件搜索路径
    add_includedirs("include")
    
    -- 链接CUDA运行时和cuBLAS
    add_links("cudart", "cublas")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end

    add_defines("CUDA_AVAILABLE")
    
    after_build(function (target)
        print("[Ok] test_linear built successfully!")
        print("Run with: xmake run test_linear")
    end)

-- check_cuda工具
target("check_cuda")
    set_kind("binary")
    set_default(false)
    
    add_files("test/check_cuda.cu")
    
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86", "compute_89")
    
    add_links("cudart")
    
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end
    
    add_defines("CUDA_AVAILABLE")

-- 定义测试目标：speed_linear (性能基准测试)
target("speed_linear")
    set_kind("binary")
    set_default(false)
    
    -- 添加测试源文件
    add_files("test/speed_linear.cu")
    
    -- 依赖tensor和layers库
    add_deps("tensor", "layers")
    
    -- CUDA相关配置
    add_rules("cuda")
    -- RTX 5070 Ti (Blackwell架构) + 向后兼容
    add_cugencodes("compute_75", "compute_80", "compute_86", "compute_89")
    
    -- 添加头文件搜索路径
    add_includedirs("include")
    
    -- 链接CUDA运行时和cuBLAS
    add_links("cudart", "cublas")
    
    -- CUDA编译选项（性能优化）
    add_cuflags("-use_fast_math")
    add_cuflags("--expt-relaxed-constexpr")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end

    add_defines("CUDA_AVAILABLE")
    
    -- Release模式下的额外优化
    if is_mode("release") then
        add_cuflags("-O3")
        if is_plat("windows") then
            add_cxxflags("/O2", "/Ob2")
        end
    end
    
    after_build(function (target)
        print("[Ok] speed_linear built successfully!")
        print("Run with: xmake run speed_linear")
        print("Tip: Use release mode for accurate benchmarks: xmake f -m release")
    end)
