OpenFV Setup
============

Below are basic instructions on how to get started with using
OpenFV. This documentation assumes that you have a basic working knowledge
of Linux.

OpenFV depends on the following packages:

#. CMake 2.8 +
#. Git
#. GCC 4.4.x +
#. Python 2.7
#. Qt4 +
#. CUDA toolkit 6 +
#. Boost Libraries
#. libtiff4 +
#. gperftools
#. Glog
#. SuiteSparse
#. Atlas
#. Eigen 3.2.2 +
#. Google Log 0.3.1 +
#. Ceres Solver
#. OpenCV 2.4.10 (3.0 not yet supported) built with Python, Qt, and CUDA


Setting up on an Ubuntu machine
-------------------------------

**Prerequisite Installations:**

*14.04*

 - ``~$ sudo apt-get install cmake build-essential libboost-all-dev qt5-default libgoogle-perftools-dev google-perftools libtiff5-dev libeigen3-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev``
 - install cuda toolkit https://developer.nvidia.com/cuda-downloads
 - install ceres http://ceres-solver.org/ceres-solver-1.10.0.tar.gz
 - ``~$ tar zxf ceres-solver-1.10.0.tar.gz && mkdir ceres-bin && cd ceres-bin``
 - ``~$ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC ../ceres-solver-1.10.0 && make && sudo make install``
 - install opencv 2.4.10 http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.10/opencv-2.4.10.zip/download
 - ``~$ unzip opencv-2.4.10.zip && cd opencv-2.4.10 && mkdir build_dir && cd build_dir && cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON ..``
 - ``~$ make && sudo make install``

*12.04*

 - ``~$ sudo apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools libeigen3-dev libatlas-base-dev libsuitesparse-dev qt4-dev-tools libtiff4-dev``
 - install glog https://github.com/google/glog
 - install cuda toolkit https://developer.nvidia.com/cuda-downloads
 - install ceres http://ceres-solver.org/ceres-solver-1.10.0.tar.gz
 - ``~$ tar zxf ceres-solver-1.10.0.tar.gz && mkdir ceres-bin && cd ceres-bin``
 - ``~$ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC -D EIGEN_INCLUDE_DIR=/usr/local/include/eigen3 ../ceres-solver-1.10.0 && make && sudo make install``
 - install opencv 2.4.10 http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.10/opencv-2.4.10.zip/download
 - (inside opencv-2.4.10): ``~$ mkdir build_dir && cd build_dir``
 - ``~$ cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON .. && make && sudo make install``

**OpenFV Installation**

 - ``~$ git clone <openfv git link>``
 - ``~$ cd openfv && ./configure && cd bin && make``
 
