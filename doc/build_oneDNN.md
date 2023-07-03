oneDNN repo: https://github.com/oneapi-src/oneDNN \
oneDNN doc: https://oneapi-src.github.io/oneDNN/

## 1 oneDNN的编译与安装
```
git clone https://github.com/oneapi-src/oneDNN.git

cd oneDNN
mkdir build
cd build

```
可选的，也可以先用默认的(默认的就cmake ..)
```
# Uncomment the following lines to build with Clang
export CC=clang
export CXX=clang++

# Uncomment the following lines to build with Intel oneAPI DPC++/C++ Compiler
export CC=icx
export CXX=icpx
```
如果选择oneAPI DPC++/C++: \
Ubuntu配置Intel oneAPI DPC++/C++ Compiler(icpc/icc): \
下载官网：https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp
\
参考CSDN教程:https://blog.csdn.net/qq_41443388/article/details/124505277 
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/89283df8-c667-47b0-b7e1-c4573e37bd3e/l_dpcpp-cpp-compiler_p_2023.1.0.46347.sh

sudo bash ./l_dpcpp-cpp-compiler_p_2023.1.0.46347.sh

echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc
或者
source /opt/intel/oneapi/setvars.sh

简单:
icx -v
gdb-oneapi -v
```


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
## 编译完后安装oneDNN
在build文件夹里，install library
```
sudo cmake --build . --target install
```
### 默认头文件是安装在/usr/local/include，库安装在/usr/local/lib

## 2. 跑example
### (好像不用，直接在build/example里直接跑)
https://zhuanlan.zhihu.com/p/546574711 \
注意primitives文件夹的： 
#include "example_utils.hpp"需要改为
#include "../example_utils.hpp"
```
g++ cpu_simple_pattern_f32.cpp -o cpu_simple_pattern_f32 -std=c++11 -I/usr/local/include -L/usr/local/lib -ldnnl

用这个
icpx -fsycl cpu_simple_pattern_f32.cpp -o cpu_simple_pattern_f32 -I/usr/local/include -L/usr/local/lib -L/opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8 -ltbb -ldnnl -lpthread
```

cd到相应example目录，如跑convolution.cpp
```
用这个：
icpx -fsycl convolution.cpp -o convolution -I/usr/local/include -L/usr/local/lib -L/opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8 -ltbb -ldnnl -lpthread

```

icpx -fsycl getting_started.cpp -o getting_started -I/usr/local/include -L/usr/local/lib -L/opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8 -ltbb -ldnnl -lpthread
### 错误：
### 1.DSO missing错误
用-L/opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8 -ltbb，是因为报错/opt/intel/oneapi/tbb/2021.9.0/env/../lib/intel64/gcc4.8/libtbb.so.12: error adding symbols: DSO missing from command line



### 2.跑convolution报错： 
error while loading shared libraries: libdnnl.so.3: cannot open shared object file: No such file or directory  
此时需要:
```
sudo find / -name "libdnnl.so.3"
```  
找到路径后export进LD_LIBRARY_PATH,哪个路径找不到就加这个路径:
```
export LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8:/usr/local/lib:/opt/intel/oneapi/compiler/2023.1.0/linux/lib:/opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
```
# 要装opencl
```
sudo apt-get install opencl-headers
sudo apt-get install ocl-icd-libopencl1

```

## 3.自己的example
```
icpx -fsycl mlp_graph.cpp -o mlp_graph -I/usr/local/include -L/usr/local/lib -L/opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8 -ltbb -ldnnl -lpthread

ldd mlp_graph
export LD_LIBRARY_PATH=/home/ubuntu/oneDNN/build/src:$LD_LIBRARY_PATH
./mlp_graph

gdb调试：
编译加上 -g
gdb ./mlp_graph

break mlp_graph.cpp:113  //设置断点
run
continue
```
