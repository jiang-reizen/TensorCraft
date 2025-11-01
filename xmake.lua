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
    add_cuflags("-Xcompiler=/utf-8")  -- 添加这行，让CUDA也使用UTF-8
    add_defines("NOMINMAX", "_CRT_SECURE_NO_WARNINGS")
end

-- 定义tensor库
target("tensor")
    set_kind("static")
    
    -- 添加源文件
    add_files("src/tensor/*.cu")
    
    -- 添加头文件搜索路径
    add_includedirs("include", {public = true})
    -- add_includedirs("include/tensor")  -- 添加这行
    
    -- CUDA相关配置
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86")
    
    -- CUDA编译选项
    add_cuflags("-use_fast_math")
    add_cuflags("--expt-relaxed-constexpr")
    
    -- 链接CUDA运行时
    add_links("cudart")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")  -- 添加这行
    end
    
    -- 定义宏
    add_defines("CUDA_AVAILABLE")
    if is_mode("debug") then
        add_defines("DEBUG")
    end

-- 定义测试目标
target("test_tensor")
    set_kind("binary")
    set_default(false)
    
    -- 添加测试源文件
    add_files("test/test_tensor.cu")
    
    -- 依赖tensor库
    add_deps("tensor")
    
    -- CUDA相关配置
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86")
    
    -- 添加头文件搜索路径
    add_includedirs("include")
    
    -- 链接CUDA运行时
    add_links("cudart")
    
    -- Windows特定
    if is_plat("windows") then
        add_cxxflags("/bigobj")
    end

    add_defines("CUDA_AVAILABLE")
    
    after_build(function (target)
        print("[Ok] Test executable built successfully!")
        print("Run with: xmake run test_tensor")
    end)

target("check_cuda")
    set_kind("binary")
    set_default(false)
    
    add_files("test/check_cuda.cu")
    
    add_rules("cuda")
    add_cugencodes("compute_75", "compute_80", "compute_86")
    
    add_links("cudart")
    
    if is_plat("windows") then
        add_cxxflags("/bigobj")
        add_cuflags("-Xcompiler=/bigobj")
    end
    
    add_defines("CUDA_AVAILABLE")

