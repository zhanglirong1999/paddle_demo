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

(用这个)
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
`cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DWITH_TESTING=ON -DWITH_INFERENCE_API_TEST=ON`  
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DWITH_INFERENCE_API_TEST=1 -DWITH_TESTING=1 -DWITH_MKL=ON -DWITH_MKLDNN=ON -DCMAKE_BUILD_TYPE=Release


编译完后TEST analisy所在目录：
```
/paddle/build/paddle/fluid/inference/tests/api
如果是develop分支，在
/paddle/build/test/cpp/inference/api
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
开启oneDNN: 设置attribute: use_mkldnn = True, 然后会到
paddle\fluid\pybind\inference_api.cc -> AnalysisPredictor.Run() -> MkldnnPreSet -> OneDNNContext init() 

其中api中可以enable mkldnn, 开启oneDnn,
其中EnableMKLDNN的最终实现应该在paddle\fluid\inference\api\paddle_pass_builder.cc ,会开启xxxx_pass，然后运行的时候执行这些IR pass,PaddlePaddle中的IR Pass是中间表示优化器.  


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
### C++ demo with Inference
https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/cpp_demo.html \
推理库安装：https://www.paddlepaddle.org.cn/inference/master/guides/install/download_lib.html#windows 
解压后的预测库paddle_inference目录(如解压后的目录名称不同，也需重命名为paddle_inference)拷贝至Paddle-Inference-Demo/c++/lib目录下

```
cd Paddle-Inference-Demo/c++/cpu/resnet50
bash compile.sh

./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams

/home/zhouziyang/lirong/Paddle-Inference-Demo/c++/cpu/resnet50/libiomp5.so

export LD_LIBRARY_PATH=/home/zhouziyang/lirong/Paddle-Inference-Demo/c++/cpu/resnet50/libiomp5.so:$LD_LIBRARY_PATH
```
----

开启IR graph可视化
```
python: config.switch_ir_debug()
c++: cfg.SwitchIrDebug()

dot -Tpng xxx.dot -o xxx.png
```


----------


python /paddle/python/paddle/fluid/tests/unittests/dygraph_to_static/test.py
python /paddle/python/paddle/fluid/tests/unittests/dygraph_to_static/test_mm.py

------

python api:如 paddle.nn.linear调用过程：
_legacy_C_ops.matmul_v2
paddle\fluid\operators\matmul_v2_op.cc
paddle\fluid\operators\matmul_op.cc

------

pdb的使用:
```
import pdb 
pdb.set_trace()

n（next）：执行下一行代码。
s（step）：进入当前行的函数或方法中。
c（continue）：继续程序执行，直到下一个断点或程序结束。
p <variable>（print）：打印变量的值。
l（list）：显示当前行的代码以及周围的代码。
q（quit）：退出调试模式。
```
-----------------
python\paddle\_legacy_C_ops.py: \
在非 Eager 模式下，用户需要先定义计算图（computation graph），然后再将数据输入到计算图中进行计算。计算图通常是由算子和变量组成的，算子表示数据的计算过程，变量表示数据的存储位置。计算图的优点是可以对整个计算过程进行优化，提高计算效率，但是它也带来了一定的限制，例如无法进行动态计算等。

在 Eager 模式下，用户可以直接使用 Python 语言进行计算，而无需先定义计算图。每个计算操作都会立即执行，结果也会立即返回。这样可以更加灵活地进行计算，支持动态计算和调试，但是可能会带来一定的计算性能损失。

在_legacy_C_ops.py中，会导入 core.ops 中的所有算子，并将它们添加到当前 Python 环境的全局命名空间中。
```
if not framework._in_eager_mode_:
    for name in dir(core.ops):
        globals()[name] = getattr(core.ops, name)
        __all__.append(name)
    _already_switch_to_eager_ = False
else:
    for name in dir(core.eager.ops.legacy):
        globals()[name] = getattr(core.eager.ops.legacy, name)
        __all__.append(name)
    _already_switch_to_eager_ = True
```
如会把matmul加入，后面如linear就调用_legacy_C_ops.matmul即可。
这些加入_legacy_C_ops的算子位于paddle\fluid\operators\xxx.cc的实现。实现了所有支持的算子

----------
跑test文件：
```
需要 docker 里build好的，在build目录下：
ctest -R test_flags_mkldnn_ops_on_off -V

如果自己写test，放在/paddle/test/mkldnn/xxx.py文件下，然后再：
vim /paddle/test/mkldnn/test_fused_matmul_mkldnn_op.py

cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DWITH_INFERENCE_API_TEST=1 -DWITH_TESTING=1 -DWITH_MKL=ON -DWITH_MKLDNN=ON -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
cd /paddle/build/python/dist
pip3.7 install -U paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

ctest -R test_fused_matmul_mkldnn_op -V

```
报错需要：
```
pip install --upgrade protobuf
pip3 install httpx
pip install opt-einsum

```

-----
开发一个kernel的流程。
PD_REGISTER_KERNEL注册kernel  --> REGISTER_OPERATOR注册op


--------

c++报错栈信息：
export FLAGS_call_stack_level=2

# paddle c++中加info：
```
#include "glog/logging.h"
LOG_FIRST_N(INFO, 10) << "New Executor is Running.";
```


## 有关动态图，静态图的一些文档：
https://www.paddlepaddle.org.cn/documentation/docs/zh/2.5rc1/dev_guides/api_contributing_guides/new_cpp_op_cn.html#span-id-paddleyaml-8-1-paddle-yaml-span 

https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html  

动态图转静态图：
https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/basic_usage_cn.html 

算子YAML文件：
https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/paddle_autogen_code.md


```
_C_ops.trace为动态图
由于目前飞桨动态图正处在重构升级阶段，所以现有算子的代码会分别有新旧动态图两个代码分支，其中 in_dygraph_mode() 表示新动态图分支（默认），_in_legacy_dygraph()为旧动态图分支，在新增算子时无需添加旧动态图分支代码。
因此：
if in_dygraph_mode():是走新版动态图
if _in_legacy_dygraph():是旧版动态图
最后才是走静态图
```
paddle.nn.Linear() 是 PaddlePaddle 中用于定义动态图全连接层的接口。 
paddle.fluid.dygraph.nn.Linear() 是 PaddlePaddle 中用于定义旧版动态图全连接层的接口。

禁用最新动态图：
```
export FLAGS_enable_eager_mode=0
禁用最新eager动态图模式
采用旧版dygraph动态图模式
```

--------
## phi和fluid kernel
有关AllOpKernels()拿到的fluid kernel是怎么来的，以及与phi kernel register的区别：

`REGISTER_OP_KERNEL(op,xxxxx)`调用了`REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE`，然后调用了`OpKernelRegistrar`--> `OpKernelRegistrarFunctor`把op加进OperatorWithKernel::AllOpKernels的map里面。

-----
一般在operator文件中，不仅register operator，还PD_REGISTER_STRUCT_KERNEL(Xxx),那么这个op的kernel大概率只有paddle自己实现的一个，没有cudnn或者mkldnn的实现

------

# 7.20
paddle\phi\api\yaml\generator\api_base.py中gen_kernel_code->gene_api_code，最后由paddle\phi\api\yaml\generator\api_gen.py去生成/paddle/paddle/phi/api/lib/api.cc中的API

```
我改了vim /paddle/paddle/phi/api/yaml/generator/api_base.py，VLOG(6)改LOG_FIRST_N(INFO, 3)
```

# 旧版动态图api路径：
paddle/fluid/pybind/eager_legacy_op_function_generator.cc  -> 
```
  // generate op function body
  auto op_function_str = paddle::string::Sprintf(OP_FUNCTION_TEMPLATE,xxxxxxxxxxx)
```
-->                  GenerateOpFunctions去创建op的代码string 
而eager_legacy_op_function_generator.cc，生成的这些配置字符串保存至/paddle/paddle/fluid/pybind/eager_legacy_op_function.cc.tmp，其中static PyObject * eager_legacy_api_matmul_v2就是matmul_v2调用的api，最终调用了matmul_v2_dygraph_function(),这个函数生成于/paddle/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions1.cc,
在matmul_v2_dygraph_function里面，调用了/paddle/paddle/fluid/imperative/tracer.cc的Tracer::TraceOp("matmul_v2")，调用到op。这是旧版动态图调用方式

# 新版动态图
paddle\fluid\eager\auto_code_generator\generator\python_c_gen.py生成了/paddle/paddle/fluid/pybind/eager_op_function.cc.tmp --->eager_api_linear调用了matmul_ad_func (在/paddle/paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.cc) ->paddle::experimental::matmul() --> api.cc里面的api


--------
试一下test_pool2d_mkldnn_op.py

paddle动静态图区别文档：
https://www.paddlepaddle.org.cn/tutorials/projectdetail/4047189

-----
inference调用：
create predictor ->CreatePaddlePredictor-> AnalysisPredictor(config)里面调用了init，init又调用了PrepareScope,PrepareProgram,CreateExecutor等,此时还会调用OptimizeInferenceProgram(),最终这个函数会遍历所有pass.ApplyImpl-> NaiveExecutor.run -> op.run -> RunImpl -> InnerGetExpectedKernelType里面设置mkldnn的lib和data_layout -》 TransOpKernelTypeToPhiKernelKey设置mkldnn backend
-----
infernence的config把model传进去，是怎么构件图的：在AnalysisPredictor::PrepareProgram中，LoadProgramDesc()中用model_path去创建inference_program_
----
graph中做pass的时候，取op:做拓扑排序，构建一个有向图中的节点之间的邻接关系，构成一个邻接表，计算图节点之间的依赖关系相关的操作
graph跑ir pass:
```
graph.reset(pass->Apply(graph.release()));
```
-----

nn.linear的静态图：layerhelper 
--》framework.py -> executor.py,链接：https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/executor.py#L802

-----
paddle.nn.xxx静态图的ir pass构造：在executor.py中Executor()-> _ExecutorCache ->_get_program_and_executor构造了graph，把program->graph->program. 

其中需要用build_strategy设置strategy，才能开启ir pass, 可以参考的例子：test_resnet.py,test_standalone_executor_fthenb_plan.py
----

-----
## test_mnist报错pool_grad:
测了一下pool backward传参的两个md的data_type，是一样的,是memory format不同，由oneDNN文档知：
https://oneapi-src.github.io/oneDNN/dev_guide_inference_and_training_aspects.html

第6和7

改：
```
    auto temp_md = dnnl::memory::desc(
        out_grad->mem_desc().get_dims(), out_grad->mem_desc().get_data_type(), OneDNNMemoryFormat::any);

//    LOG_FIRST_N(INFO, 10) <<out_grad->mem_desc().get_dims();
    this->AcquireBackwardPrimitiveDescriptor(
        pooling_type == "max"
            ? dnnl::algorithm::pooling_max
            : (exclusive ? dnnl::algorithm::pooling_avg_exclude_padding
                         : dnnl::algorithm::pooling_avg_include_padding),
        diff_src_md,
     //   out_grad->mem_desc(),
        temp_md,
        copied_strides,
        copied_kernel_size,
        dilation,
        onednn_paddings[0],
        onednn_paddings[1]);
    LOG_FIRST_N(INFO, 10) << "kkkkkkkkkkkkk";
  }
```

----


