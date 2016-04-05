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

   * Some versions may not be compatable. In such a case, please let us know



Set-up on an Ubuntu Machine
===========================

14.04
^^^^^

.. code-block:: bash

    $ sudo apt-get install qt5-default libtiff5-dev libgoogle-glog-dev
   

12.04
^^^^^

.. code-block:: bash

    $ sudo apt-get install qt4-dev-tools libtiff4-dev

* install `glog
  <https://google-glog.googlecode.com/svn/trunk/INSTALL>`_



General Setup
^^^^^^^^^^^^^
.. code-block:: bash

    $ sudo apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools libeigen3-dev libatlas-base-dev libsuitesparse-dev  

* install `ceres <http://ceres-solver.org/building.html>`_

After installing ceres, run the following commands in your build directory

.. code-block:: bash

    $ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC ../ceres-solver-1.10.0 && make && sudo make install

* install `cuda toolkit <http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`_

* install `opencv 2.4.10 <http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html>`_

After installing opencv 2.4.10, change into the directory and run the following commands

.. code-block:: bash

    $ mkdir build_dir && cd build_dir && cmake -D CUDA_GENERATION=Kepler -D WITH_QT=ON ..
    $ make && sudo make install


OpenFV Installation
^^^^^^^^^^^^^^^^^^^

*Cloning Repository*

.. code-block:: bash

    $ git clone <openfv git link>
    
*Basic Installation*

.. code-block:: bash

    $ cd openfv && ./configure && cd bin && make

*Install with Python Wrappers* **(EXPERIMENTAL)**

.. code-block :: bash

    $ cd openfv && mkdir bin && cd bin
    $ cmake -D BUILD_PYTHON=ON ..
    $ make
