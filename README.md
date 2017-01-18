OpenFV Setup
============

<p>Below are basic instructions on how to get started with using
OpenFV. This documentation assumes that you have a basic working knowledge
of Linux.

OpenFV depends on the following packages:</p>

1. CMake 2.8 or higher
2. Git
3. GCC 4.4.x or higher
4. Python 2.7
5. Qt4 +
6. CUDA toolkit 6 +
7. Boost Libraries, libtiff4 +, gperftools
8. Glog, Google Log 0.3.1 +
9. Ceres Solver, Eigen 3.2.2 +, Atlas, SuiteSparse
10. OpenCV 2.4.10 or earlier (3.0+ not yet supported) built with Python, Qt, and CUDA

*   Some versions may not be compatable. In such a case, please let us know



Set-up on an Ubuntu Machine
===========================

### 14.04


<pre><code>$ sudo apt-get install qt5-default libtiff5-dev libgoogle-glog-dev </code></pre>
   

### 12.04



<pre><code>$ sudo apt-get install qt4-dev-tools libtiff4-dev </code></pre>

* install <a href="https://google-glog.googlecode.com/svn/trunk/INSTALL">glog</a>


### General Setup

<pre><code>$ sudo apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools libeigen3-dev libatlas-base-dev libsuitesparse-dev </code></pre> 

* install <a href="http://ceres-solver.org/building.html">ceres</a>

<p>After installing ceres, run the following commands in your build directory</p>

<pre><code>$ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC ../ceres-solver-1.10.0 && make && sudo make install </code></pre>

* install <a href="http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf">cuda toolkit</a>

* install <a href="http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html">opencv 2.4.10</a>
<p>After installing opencv 2.4.10, change into the directory and run the following commands</p>



<pre><code>$ mkdir build_dir && cd build_dir && cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON ..
$ make && sudo make install
</code></pre>


### OpenFV Installation


*Cloning Repository*


<pre><code>$ git clone (openfv git link) </code></pre>
    
*Basic Installation*

<pre><code>$ cd openfv && ./configure && cd bin && make</code></pre>

*Install with Python Wrappers* **(EXPERIMENTAL)**

<pre><code>$ cd openfv && mkdir bin && cd bin
$ cmake -D BUILD_PYTHON=ON ..
$ make
</code></pre>



Roadmap
=======

### Phase 1
- Complete Python bindings for all C++ functionality
- Add particle field rendering engine
- Add tomo reconstruction and window deformation enabled PIV code (MATLAB)

### Phase 2
- Add tests that ensure functionality is not breaking due to updates
- Remove MATLAB code and call underlying algorithms (in Fortran) from Python
- Eliminate use of Intel MKL (in Fortran code) and replace with FFTW
- Start establishing a framework for effective benchmarking and comparison of different algorithms using the exact same data / inputs

### Phase 3
- Streamline execution pipeline so that reconstruction via multiple techniques can seamlessly be bound to multiple PIV algorithms for comparison purposes
- Develope API for researchers to hook in custom reconstruction and PIV algorithms for benchmarking