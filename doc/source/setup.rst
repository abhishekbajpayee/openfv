OpenFV Setup
============

Below are basic instructions on how to get started with using
OpenFV. This documentation assumes that you have a basic working knowledge
of Linux.

OpenFV depends on the following packages:

#. CMake 2.8 or higher
#. Git
#. GCC 4.4.x or higher
#. Python 2.7
#. Qt4 +
#. CUDA toolkit 6 +
#. Boost Libraries, libtiff4 +, gperftools
#. Glog, Google Log 0.3.1 +
#. Ceres Solver, Eigen 3.2.2 +, Atlas, SuiteSparse
#. OpenCV 2.4.10 or earlier (3.0+ not yet supported) built with Python, Qt, and CUDA

   * Some versions may not be compatable. In such a case, please notify us.



Setting up on an Ubuntu machine
-------------------------------

**Prerequisite Installations:**

*14.04*

.. code-block:: bash

    $ sudo apt-get install cmake build-essential libboost-all-dev qt5-default libgoogle-perftools-dev google-perftools libtiff5-dev libeigen3-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev


* install `cuda toolkit <http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`_
* install `ceres <http://ceres-solver.org/building.html>`_
After installing ceres, run the following commands in your build directory

.. code-block:: bash 

   $ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC ../ceres-solver-1.10.0 && make && sudo make install
   

* install `opencv 2.4.10 <http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html>`_


After installing opencv 2.4.10, enter the directory and run the following commands

.. code-block:: bash

    $ mkdir build_dir && cd build_dir && cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON ..
    $ make && sudo make install

*12.04*

.. code-block:: bash

    $ sudo apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools libeigen3-dev libatlas-base-dev libsuitesparse-dev qt4-dev-tools libtiff4-dev


* install `glog <https://google-glog.googlecode.com/svn/trunk/INSTALL>`_
* install `cuda toolkit <https://developer.nvidia.com/cuda-downloadshttp://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`_
* install `ceres <http://ceres-solver.org/building.html>`_
.. code-block:: console

    $ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC -D EIGEN_INCLUDE_DIR=/usr/local/include/eigen3 ../ceres-solver-1.10.0 && make && sudo make install

* install `opencv 2.4.10 <http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html>`_


After installing opencv 2.4.10, enter the directory and run the following commands

.. code-block:: bash

    $ mkdir build_dir && cd build_dir && cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON ..
    $ make && sudo make install

**OpenFV Installation**

- ``~$ git clone <openfv git link>``
- ``~$ cd openfv && ./configure && cd bin && make``
 
