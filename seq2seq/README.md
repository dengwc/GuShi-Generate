## Seq2Seq part

using [CNN](https://github.com/Oneplus/cnn.git) library to build it . Example file `encdec.cc` is taken into account .

## Compile

1. Download EIGEN3

    可以直接从[项目网址](https://bitbucket.org/eigen/eigen)下载已经Release的包。
    
    如果想要克隆，EIGEN3是在bitbucket上，默认支持水银(Mercurial)分布式版本控制工具进行管理；如果要使用Git克隆当前版本，参见[Clone a repository](https://confluence.atlassian.com/bitbucket/clone-a-repository-223217891.html) , 或者直接从GitHub上克隆EIGEN3的[镜像](https://github.com/OPM/eigen3)

2. Download & Install `CMake`

    [CMake](https://cmake.org/)

3. Install Boost 
    
    [Boost](http://www.boost.org/) , 无论在Linux或者Windows上都是比较好安装的。只是Windows上似乎只能编译，比较耗时。Linux上试试直接用包管理工具安装。

4. 初始化cnn子模块
    
    ```shell
    git submodule init
    git submodule update
    ```
5. 编译

     ```CMAKE
    mkdir build
    cd build 
    cmake .. -DBOOST_ROOT=/absolutepath/to/boost -DEIGEN3_INCLUDE_DIR=/absolutepath/to/eigen3 -DBoost_USE_STATIC_LIBS=On
    ```

## 说明

CNN库不支持GPU，在层数较多、节点数量多时也是比较慢的。

目前已经由LSTM改为使用RNN来做ENCDEC。

按照默认参数训练1K的诗（五言绝句）需要约4分钟。
