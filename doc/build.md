oneDNN repo: https://github.com/oneapi-src/oneDNN \
oneDNN doc: https://oneapi-src.github.io/oneDNN/

## 1 oneDNN build and install 
```
git clone https://github.com/oneapi-src/oneDNN.git

cd oneDNN
mkdir build
cd build

```
optional(default is use cmake ..)
```
# Uncomment the following lines to build with Clang
export CC=clang
export CXX=clang++

# Uncomment the following lines to build with Intel oneAPI DPC++/C++ Compiler
export CC=icx
export CXX=icpx
```
## oneAPI DPC++/C++ Install: 
Ubuntu download Intel oneAPI DPC++/C++ Compiler(icpc/icc): \
下载官网：https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp
\
CSDN Guide: https://blog.csdn.net/qq_41443388/article/details/124505277 
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/89283df8-c667-47b0-b7e1-c4573e37bd3e/l_dpcpp-cpp-compiler_p_2023.1.0.46347.sh

sudo bash ./l_dpcpp-cpp-compiler_p_2023.1.0.46347.sh

echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc
or
source /opt/intel/oneapi/setvars.sh

icx -v
gdb-oneapi -v
```
## Build oneDNN 

```
export CC=icx
export CXX=icpx

cmake .. \
          -DDNNL_CPU_RUNTIME=SYCL \
          -DDNNL_GPU_RUNTIME=SYCL \
          -DDNNL_BUILD_EXAMPLES=ON
make -j12

cd example
直接跑
```
## Install oneDNN
In build file，install library
```
sudo cmake --build . --target install
```
### 默认头文件是安装在/usr/local/include，库安装在/usr/local/lib

## 2. Run example
```
cd build
./examples/graph-simple-pattern-f32-cpp
```
If has error like: libdnnl.so.3 no such file or dictionary:
```
ldd ./examples/graph-simple-pattern-f32-cpp

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/home/ubuntu/intel/oneDNN/build/src/libdnnl.so.3

export LD_LIBRARY_PATH=/home/ubuntu/intel/oneDNN/build/src/libdnnl.so.3
```
## 3. Errors in graph API
```
    /// create a new engine and stream
    allocator alloc {};
    dnnl::engine eng =make_engine_with_allocator(ekind, 0, alloc);
    dnnl::stream strm {eng};
```
There is an error:
```
terminate called after throwing an instance of 'dnnl::error'
what():  could not create allocator
Aborted (core dumped)
```
Reason: if use SYCL runtime, use `dnnl_graph_sycl.hpp` not `dnnl_graph.hpp`.
```
allocator alloc {dnnl::graph::sycl_interop::make_allocator(
            sycl_malloc_wrapper,
            sycl_free_wrapper)};
            
dnnl::engine eng =make_engine_with_allocator(ekind, 0, alloc);
dnnl::stream strm {eng};
```

## 性能: 
1. sudo perf record -e cycles ./examples/primitives-example-mlp-cpp 
perf report


2. export DNNL_VERBOSE=1 \
./examples/primitives-example-mlp-fuse-cpp
```
ubuntu@VM-12-15-ubuntu:~/intel/oneDNN/build2$ ./examples/primitives-example-mlp-cpp
onednn_verbose,info,oneDNN v3.2.0 (commit 2a3fbc8a43e553af97a4936cb6f39e5935af5e72)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:4
onednn_verbose,info,cpu,isa:Intel AVX2
onednn_verbose,info,gpu,runtime:none
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,exec,cpu,matmul,gemm:jit:f32,undef,src_f32::blocked:abc::f0 wei_f32::blocked:abc::f0 bia_f32::blocked:abc::f0_mask4 dst_f32::blocked:abc::f0,,,1x12800x128:1x128x64,39.0911
onednn_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:abc::f0 diff_undef::undef:::,,alg:eltwise_relu alpha:1.2 beta:0,1x12800x64,13.1912
onednn_verbose,exec,cpu,matmul,gemm:jit:f32,undef,src_f32::blocked:abc::f0 wei_f32::blocked:abc::f0 bia_f32::blocked:abc::f0_mask4 dst_f32::blocked:abc::f0,,,1x12800x64:1x64x32,26.54
onednn_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:abc::f0 diff_undef::undef:::,,alg:eltwise_relu alpha:0.7 beta:0,1x12800x32,25.1069
Elapsed time on double mlp without post-ops: 179 ms
Example passed on CPU.
ubuntu@VM-12-15-ubuntu:~/intel/oneDNN/build2$ ./examples/primitives-example-mlp-fuse-cpp
onednn_verbose,info,oneDNN v3.2.0 (commit 2a3fbc8a43e553af97a4936cb6f39e5935af5e72)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:4
onednn_verbose,info,cpu,isa:Intel AVX2
onednn_verbose,info,gpu,runtime:none
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,exec,cpu,matmul,gemm:jit:f32,undef,src_f32::blocked:abc::f0 wei_f32::blocked:abc::f0 bia_f32::blocked:abc::f0_mask4 dst_f32::blocked:abc::f0,attr-post-ops:eltwise_relu:1.2 ,,1x12800x128:1x128x64,18.0881
onednn_verbose,exec,cpu,matmul,gemm:jit:f32,undef,src_f32::blocked:abc::f0 wei_f32::blocked:abc::f0 bia_f32::blocked:abc::f0_mask4 dst_f32::blocked:abc::f0,attr-post-ops:eltwise_relu:0.7 ,,1x12800x64:1x64x32,1.01807
Elapsed time on double mlp with post-ops fuse ReLU: 43 ms
Example passed on CPU.

```
## stable performance配置：
```
You can try it with numactl tools on linux:

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

export OMP_NUM_THREADS=N # your-cpu-physical-core-number

export KMP_BLOCKTIME=1

numactl --physcpubind=0-N-1 --membind=0 ./your-app-name –your-defined-args
```

# Paddle install
https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html#compile_from_host

其中：(如果使用docker就不需要，之后只需`sudo docker exec -it paddle-test /bin/bash`)
```
或者
docker start paddle-test
docker attach paddle-test
```

注意一定要 develop分支： \
`cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DWITH_TESTING=ON`  
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DWITH_TESTING=ON \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON \
    -DWITH_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release
编译完后所在目录：
```
/paddle/build/paddle/fluid/inference/tests/api
```
记得要把model放在/model, 参数放在/params 

然后`./test_analyzer_image_classification`


编译完成后运行单元测试：
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html#yunxingdanyuanceshi

paddle算子介绍：
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html

# 运行paddle的test
安装gtest: https://blog.csdn.net/Swallow_he/article/details/120065786
```
apt-get install libgtest-dev
apt-get install libgflags-dev
apt-get install libgoogle-glog-dev
apt-get install protobuf-compiler libprotobuf-dev

cd /paddle/build/test/cpp/inference/api
sudo find / -name port_def.inc
export CPLUS_INCLUDE_PATH=/paddle/third_party/protobuf/src:/paddle/build/third_party/install/protobuf/include:/paddle/third_party/eigen3:/paddle/build:/paddle:/paddle/build/third_party/install/xxhash/include:/paddle/third_party/dlpack/include:/paddle/build/paddle_inference_install_dir/third_party/threadpool:/paddle/build/paddle_inference_install_dir/paddle/include

g++ -std=c++14 analyzer_image_classification_tester.cc -o analyzer_image_classification_tester -lgtest -lpthread -lglog -lgflags
```

## If don't use docker install paddle
```
vim ~/.bashrc
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV_ARGS='--no-site-packages'
export VIRTUALENVWRAPPER_PYTHON=/home/ubuntu/anaconda3/envs/py37/bin/python
source /home/ubuntu/anaconda3/envs/py37/bin/virtualenvwrapper.sh

export PYTHON_LIBRARY=/home/ubuntu/anaconda3/envs/py37/lib/libpython3.so
export PYTHON_INCLUDE_DIRS=/home/ubuntu/anaconda3/include/python3.7m
export PATH=/home/ubuntu/anaconda3/envs/py37/bin:$PATH
------------------------------
export PYTHON_LIBRARY=/anaconda/envs/py37/lib/libpython3.so
export PYTHON_INCLUDE_DIRS=/anaconda/envs/py37/include/python3.7m

export PATH=/anaconda/envs/py37/bin:$PATH
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV_ARGS='--no-site-packages'
export VIRTUALENVWRAPPER_PYTHON=/anaconda/envs/py37/bin/python
source /anaconda/envs/py37/bin/virtualenvwrapper.sh

source ~/.bashrc

版本问题：
pip install virtualenv==16.7.9

workon paddle-venv

```

更新cmake: 
https://zhuanlan.zhihu.com/p/519732843

```
cmake .. -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON \
    -DWITH_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release

```

磁盘有用的命令：
```
df -h
du -h --max-depth=1
```


```
cmake .. -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DCMAKE_BUILD_TYPE=Debug -DWITH_INFERENCE_API_TEST=1 -DWITH_TESTING=1 -DWITH_NCCL=OFF -DWITH_RCCL=OFF -DWITH_XPU_BKCL=OFF -DWITH_GLOO=OFF -DWITH_CUSTOM_DEVICE=OFF -DWITH_CINN=OFF
```

## Paddle源码阅读
Cmake目录里存放的是源码编译之间的链式结构 \
Paddle目录里存放的是Paddle底层C++以及CUDA的实现代码 \
Python目录里存放的是Python接口的实现以及调用方式(paddle的上层python api就在这里，通过调用_C_ops-> core -> _get_phi_kernel_name -> phi::TransToPhiKernelName) 


------
开启oneDNN: 设置attribute: use_mkldnn = True, 然后会到paddle\fluid\inference\api\analysis_predictor.cc, MkldnnPreSet, 然后就调用到了phi::OneDNNContext里的东西。而这个`analysis_predictor`会在paddle\fluid\jit的PredictorEngine的engine中调用(`predictor_.reset(new AnalysisPredictor(config));`)，加入到paddle的jit和layer中。在：
```
     layer.SetEngine(
          info->FunctionName(),
          utils::MakeEngine<PredictorEngine>(info, params_dict, place));
```
把engine加入到layer中。 

在paddle\phi\core\flags.cc中：
```
 * FLAGS_jit_engine_type == New, using InterpreterEngine by default
 * FLAGS_jit_engine_type == Predictor, using inference Predictor by default
 */
PHI_DEFINE_EXPORTED_string(jit_engine_type,
                           "Predictor",
                           "Choose default function type in JitLayer.");
```
默认就是启用了`PredictorEngine`  

因此调用过程: jit -> layer -> setEngine ->默认PredictorEngine -> AnalysisPredictor -> 发现attr: use_mkldnn,调用MkldnnPreSet, 调用到oneDNN的context 

这个：
paddle\fluid\pybind\inference_api.cc -> AnalysisPredictor.Run() -> MkldnnPreSet -> OneDNNContext init() 

其中api中可以enable mkldnn, 开启oneDnn,
其中EnableMKLDNN的最终实现应该在paddle\fluid\inference\api\paddle_pass_builder.cc ,会开启xxxx_pass，然后运行的时候执行这些IR pass,PaddlePaddle中的IR Pass是中间表示优化器. 
op Run: void OperatorBase::Run -> RunImpl -> BuildPhiKernelContext(relate to OneDNNContext, onednn backend) -> Build input and output -> 把Attr添加到phi_kernel_context中

关于kernel：
```
      phi_kernel_.reset(
          new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
              phi_kernel_name, phi_kernel_key)));
```
通过phi_kernel_name, phi_kernel_key创建对应kernel



## 算子开发:
yaml -> InferMeta -> Kernel -> 封装 Python API  
其中yaml配置的算子名称，在paddle/fluid/operators/xxx中， 将配置文件中的算子信息注册到框架内供执行器调度

## windows 安装paddle
nanja的安装：https://www.jianshu.com/p/2f93fd6a64c9


## Paddle Inference 开启oneDNN
### python demo
https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/python_demo.html 
https://www.paddlepaddle.org.cn/inference/master/guides/x86_cpu_infer/paddle_x86_cpu.html
```
python python_demo.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --batch_size 2
```
### C++ demo
https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/cpp_demo.html 

