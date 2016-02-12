OpenFV Setup
============

Below are basic instructions on how to get started with using
OpenFV. This documentation assumes that you have a basic working knowledge
of Linux.

OpenFV depends on the following packages:

<<<<<<< HEAD
#. CMake 2.8 or higher
#. Git
#. GCC 4.4.x or higher
=======
#. CMake 2.8 +
#. Git
#. GCC 4.4.x +
>>>>>>> d8dfed300a176d86db37557a45660dbdd4f85820
#. Python 2.7
#. Qt4 +
#. CUDA toolkit 6 +
#. Boost Libraries
#. libtiff4 +
#. gperftools
<<<<<<< HEAD
#. Glog, Google Log 0.3.1 +
#. SuiteSparse
#. Atlas
#. Eigen 3.2.2 +
#. Ceres Solver
#. OpenCV 2.4.10 or earlier (3.0+ not yet supported) built with Python, Qt, and CUDA

   * Some versions may not be compatable. In such a case, please notify us.
=======
#. Glog
#. SuiteSparse
#. Atlas
#. Eigen 3.2.2 +
#. Google Log 0.3.1 +
#. Ceres Solver
#. OpenCV 2.4.10 (3.0 not yet supported) built with Python, Qt, and CUDA
>>>>>>> d8dfed300a176d86db37557a45660dbdd4f85820


Setting up on an Ubuntu machine
-------------------------------

**Prerequisite Installations:**

*14.04*

 - ``~$ sudo apt-get install cmake build-essential libboost-all-dev qt5-default libgoogle-perftools-dev google-perftools libtiff5-dev libeigen3-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev``
 - install `cuda toolkit <http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`_
 - install `ceres <http://ceres-solver.org/building.html>`_
 - ``~$ tar zxf ceres-solver-1.10.0.tar.gz && mkdir ceres-bin && cd ceres-bin``
 - ``~$ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC ../ceres-solver-1.10.0 && make && sudo make install``
 - install `opencv 2.4.10 <http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html>`_
 - ``~$ unzip opencv-2.4.10.zip && cd opencv-2.4.10 && mkdir build_dir && cd build_dir && cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON ..``
 - ``~$ make && sudo make install``

*12.04*

 - ``~$ sudo apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools libeigen3-dev libatlas-base-dev libsuitesparse-dev qt4-dev-tools libtiff4-dev``
 - install `glog <https://google-glog.googlecode.com/svn/trunk/INSTALL>`_
 - install `cuda toolkit <https://developer.nvidia.com/cuda-downloadshttp://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`_
 - install `ceres <http://ceres-solver.org/building.html>`_
 - ``~$ tar zxf ceres-solver-1.10.0.tar.gz && mkdir ceres-bin && cd ceres-bin``
 - ``~$ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC -D EIGEN_INCLUDE_DIR=/usr/local/include/eigen3 ../ceres-solver-1.10.0 && make && sudo make install``
 - install `opencv 2.4.10 <http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html>`_
 - (inside opencv-2.4.10): ``~$ mkdir build_dir && cd build_dir``
 - ``~$ cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON .. && make && sudo make install``

**OpenFV Installation**

 - ``~$ git clone <openfv git link>``
 - ``~$ cd openfv && ./configure && cd bin && make``
 
