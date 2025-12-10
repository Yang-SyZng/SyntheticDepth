# Installation
## 系统环境设置
> # 仅Linux
```bash
1、配置CUDA环境
https://developer.nvidia.com/cuda-11-6-0-download-archive
(50系显卡CUDA环境设置)
https://developer.nvidia.com/cuda-12-8-1-download-archive

2、conda源设置（不建议，有能力还是直接翻墙）：
`
nano ~/.condarc
`
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/r
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud
  bioconda: https://mirrors.ustc.edu.cn/anaconda/cloud
```
## python 环境设置（50系显卡见后面）
```bash
# clone repo
# if you use miniforge(mamba) and anaconda(conda) 
1. mamba(conda) create -n gs-depth python==3.7.13

2. conda activate gs-depth

3. mamba(conda) install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

4.❗(一定要做的降级安装) mamba install mkl==2024.0

5. pip install submodules/diff-gaussian-rasterization-sig-depth

6. pip install submodules/simple-knn

7. pip install laspy
   pip install tqdm
   pip install open3d
......其他软件包，按照报错依次补全
```
### 50系显卡 python 环境设置
```bash
# clone repo
# if you use miniforge(mamba) and anaconda(conda) 
1. mamba(conda) create -n gs-depth python=3.9

2. conda activate gs-depth

3. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

4. pip install submodules/diff-gaussian-rasterization-sig-depth

5. pip install submodules/simple-knn

6. pip install laspy
   pip install tqdm
   pip install open3d
......其他软件包，按照报错依次补全
```
## 运行
```bash
python generate.py -s path/to/data --data_device cpu
```

## 遇到的问题
```bash
问题1：安装submodules/diff-gaussian-rasterization-sig-depth、pip install submodules/simple-knn 提示gcc、g++版本过高

解答1：1. sudo apt install gcc-9 g++-9
2. export CC=gcc-9
export CXX=g++-9


问题2：安装submodules/diff-gaussian-rasterization-sig-depth、pip install submodules/simple-knn 提示找不到cuda

回答2：nano ~/.bashr 查看是否有以下几行
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export CUDA_HOME=/usr/local/cuda-11.6
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 
export PATH=$CUDA_HOME/bin:$PATH


问题3：其他报错，如"ImportError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory"

回答3：卸载旧版本，尝试安装
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```
### 50系显卡当中遇到的问题
问题1：pip install submodules/diff-gaussian-rasterization-sig-depth 安装失败
```bash
xxx@xxx: your/path/synthetic_depth$ pip install submodules/diff-gaussian-rasterization-sig-depth
Processing ./submodules/diff-gaussian-rasterization-sig-depth
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: diff_gaussian_rasterization
  Building wheel for diff_gaussian_rasterization (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [71 lines of output]
      running bdist_wheel
      you/path/lib/python3.9/site-packages/torch/utils/cpp_extension.py:576: UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.
        warnings.warn(msg.format('we could not find ninja.'))
      running build
      running build_py
      creating build/lib.linux-x86_64-cpython-39/diff_gaussian_rasterization
      copying diff_gaussian_rasterization/__init__.py -> build/lib.linux-x86_64-cpython-39/diff_gaussian_rasterization
      running build_ext
      you/path/lib/python3.9/site-packages/torch/utils/cpp_extension.py:490: UserWarning: There are no g++ version bounds defined for CUDA version 12.8
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
      building 'diff_gaussian_rasterization._C' extension
      creating build/temp.linux-x86_64-cpython-39/cuda_rasterizer
      you/path/lib/python3.9/site-packages/torch/utils/cpp_extension.py:2356: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
      If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
        warnings.warn(
      /usr/local/cuda-12.8/bin/nvcc -Iyou/path/lib/python3.9/site-packages/torch/include -Iyou/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -Iyou/path/include/python3.9 -c cuda_rasterizer/backward.cu -o build/temp.linux-x86_64-cpython-39/cuda_rasterizer/backward.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -I/home/zty/projections/GTLR-GS/tools/synthetic_depth/submodules/diff-gaussian-rasterization-sig-depth/third_party/glm/ -O3 -std=c++14 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120
      cuda_rasterizer/auxiliary.h(156): warning #177-D: variable "p_proj" was declared but never referenced
         float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
                ^
      
      Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"
      
      g++ -pthread -B you/path/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem you/path/include -Iyou/path/include -fPIC -O2 -isystem you/path/include -fPIC -Iyou/path/lib/python3.9/site-packages/torch/include -Iyou/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -Iyou/path/include/python3.9 -c ext.cpp -o build/temp.linux-x86_64-cpython-39/ext.o -O2 -std=c++17 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1
      /usr/local/cuda-12.8/bin/nvcc -Iyou/path/lib/python3.9/site-packages/torch/include -Iyou/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -Iyou/path/include/python3.9 -c rasterize_points.cu -o build/temp.linux-x86_64-cpython-39/rasterize_points.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -I/home/zty/projections/GTLR-GS/tools/synthetic_depth/submodules/diff-gaussian-rasterization-sig-depth/third_party/glm/ -O3 -std=c++14 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120
      In file included from you/path/lib/python3.9/site-packages/torch/include/torch/extension.h:5,
                       from rasterize_points.cu:13:
      you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/all.h:4:2: error: #error C++17 or later compatible compiler is required to use PyTorch.
          4 | #error C++17 or later compatible compiler is required to use PyTorch.
            |  ^~~~~
      In file included from you/path/lib/python3.9/site-packages/torch/include/ATen/core/TensorBase.h:14,
                       from you/path/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:38,
                       from you/path/lib/python3.9/site-packages/torch/include/ATen/core/Tensor.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/autograd/function_hook.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/autograd/cpp_hook.h:2,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/autograd/variable.h:6,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/autograd/autograd.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/autograd.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/all.h:7,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/extension.h:5,
                       from rasterize_points.cu:13:
      you/path/lib/python3.9/site-packages/torch/include/c10/util/C++17.h:24:2: error: #error You need C++17 to compile PyTorch
         24 | #error You need C++17 to compile PyTorch
            |  ^~~~~
      In file included from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/data/dataloader_options.h:4,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/data/dataloader/base.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/data/dataloader/stateful.h:4,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/data/dataloader.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/data.h:3,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/all.h:9,
                       from you/path/lib/python3.9/site-packages/torch/include/torch/extension.h:5,
                       from rasterize_points.cu:13:
      you/path/lib/python3.9/site-packages/torch/include/ATen/ATen.h:4:2: error: #error C++17 or later compatible compiler is required to use ATen.
          4 | #error C++17 or later compatible compiler is required to use ATen.
            |  ^~~~~
      error: command '/usr/local/cuda-12.8/bin/nvcc' failed with exit code 1
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for diff_gaussian_rasterization
  Running setup.py clean for diff_gaussian_rasterization
Failed to build diff_gaussian_rasterization
ERROR: Failed to build installable wheels for some pyproject.toml based projects (diff_gaussian_rasterization)
```
回答1：
> #error You need C++17 to compile PyTorch

安装错误主要是由于 编译器不支持 C++17 标准，而 PyTorch 和相关 CUDA 扩展现在 必须使用支持 C++17 的编译器。

打开 submodules/diff-gaussian-rasterization-sig-depth/setup.py，改成"-std=c++17"
```python
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++17'],
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"), '-O3', '-std=c++17']
                })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```
重新安装

问题2：pip install submodules/simple-knn 安装失败
```bash
xxx@xxx:your/path/synthetic_depth$ pip install submodules/simple-knn
Processing ./submodules/simple-knn
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: simple_knn
  Building wheel for simple_knn (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [28 lines of output]
      running bdist_wheel
      you/path/lib/python3.9/site-packages/torch/utils/cpp_extension.py:576: UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.
        warnings.warn(msg.format('we could not find ninja.'))
      running build
      running build_ext
      you/path/lib/python3.9/site-packages/torch/utils/cpp_extension.py:490: UserWarning: There are no g++ version bounds defined for CUDA version 12.8
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
      building 'simple_knn._C' extension
      creating build/temp.linux-x86_64-cpython-39
      g++ -pthread -B you/path/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem you/path/include -Iyou/path/include -fPIC -O2 -isystem you/path/include -fPIC -Iyou/path/lib/python3.9/site-packages/torch/include -Iyou/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -Iyou/path/include/python3.9 -c ext.cpp -o build/temp.linux-x86_64-cpython-39/ext.o -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17
      you/path/lib/python3.9/site-packages/torch/utils/cpp_extension.py:2356: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
      If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
        warnings.warn(
      /usr/local/cuda-12.8/bin/nvcc -Iyou/path/lib/python3.9/site-packages/torch/include -Iyou/path/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -Iyou/path/include/python3.9 -c simple_knn.cu -o build/temp.linux-x86_64-cpython-39/simple_knn.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120 -std=c++17
      simple_knn.cu:23: warning: "__CUDACC__" redefined
         23 | #define __CUDACC__
            |
      <command-line>: note: this is the location of the previous definition
      simple_knn.cu(90): error: identifier "FLT_MAX" is undefined
          me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
                      ^
      
      simple_knn.cu(154): error: identifier "FLT_MAX" is undefined
         float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
                           ^
      
      2 errors detected in the compilation of "simple_knn.cu".
      error: command '/usr/local/cuda-12.8/bin/nvcc' failed with exit code 2
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for simple_knn
  Running setup.py clean for simple_knn
Failed to build simple_knn
ERROR: Failed to build installable wheels for some pyproject.toml based projects (simple_knn)
```
回答2：打开submodules/simple-knn/simple_knn.cu
添加：
> #include <float.h>

重新安装