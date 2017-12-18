Set-up on Cloud Instances (AWS EC2)
===================================

*(experimental)*

We realize that code libraries can be painful to install and not
everyone has access to a Linux machine. This is why we have packaged 
OpenFV to be easily set up on AWS EC2 instances inside docker
containers. Just follow the following simple steps to get started:


#. Create an AWS account
#. Create an EC2 instance using the `OpenFV AMI <#>`. This AMI has all
   the requirements pre-installed. Also, if you pick an instance type
   that has a connected GPU(s) (we recommend the `p2 instances <https://aws.amazon.com/ec2/instance-types/p2/>`), this
   AMI will be able to communicate with the GPU.
#. SSH into your instance. If your instance does not have a GPU, run:

   .. code-block :: bash
       
       $ docker pull openfv/openfv
       $ docker run -it --rm -p 80:8888 openfv/openfv

   If your instance has a GPU, run:

   .. code-block :: bash
       
       $ docker pull openfv/openfv-cuda
       $ nvidia-docker run -it --rm -p 80:8888 openfv/openfv-cuda

#. Now access a IPython Notebook / Jupyter server that runs inside
   your EC2 instance by visiting the public DNS of your instance. Now
   you can create Python notebooks and use the OpenFV Python API or
   write C++ executables and run them via the terminal.

Set-up on an Ubuntu Machine
===========================

Below are basic instructions on how to get started with using
OpenFV. This documentation assumes that you have a basic working knowledge
of Linux.

OpenFV requires the following packages:

#. CMake 2.8 or higher
#. Git
#. GCC 4.4.x or higher
#. Python 2.7
#. Boost Libraries, libtiff4 +, gperftools
#. Glog, Google Log 0.3.1 +
#. Ceres Solver, Eigen 3.2.2 +, Atlas, SuiteSparse
#. yaml-cpp
#. OpenCV 2.4.10 (3.0+ not yet supported) built with Python,
   [optional: Qt and CUDA (if you have a GPU on your machine)]

   * Some versions may not be compatable. In such a case, please let
     us know.

The following packages are optional for OpenFV:

#. Qt4 +
#. CUDA toolkit 7.5 preferred (6.0+ should work, 8.0 not supported)

   * If Qt is no installed, any live view functions will not work.

14.04
^^^^^

.. code-block:: bash

    $ sudo apt-get install libtiff5-dev libgoogle-glog-dev
    $ sudo apt-get install qt5-default
   

12.04
^^^^^

.. code-block:: bash

    $ sudo apt-get install libtiff4-dev
    $ sudo apt-get install qt4-dev-tools

* install `glog
  <https://google-glog.googlecode.com/svn/trunk/INSTALL>`_



General Setup
^^^^^^^^^^^^^
.. code-block:: bash

    $ sudo apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools libeigen3-dev libatlas-base-dev libsuitesparse-dev libyaml-cpp-dev

* install `ceres <http://ceres-solver.org/building.html>`_

After installing ceres, run the following commands in your build directory

.. code-block:: bash

    $ cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC ../ceres-solver-1.10.0 && make && sudo make install

* install `CUDA Toolkit <http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`_

* install `OpenCV 2.4.10 <http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html>`_

After installing opencv 2.4.10, change into the directory and run the following commands:

.. code-block:: bash

    $ mkdir bin && cd bin && cmake -D CUDA_GENERATION=Kepler [-D WITH_QT=ON] ..
    $ make

or if you are building without CUDA:

.. code-block:: bash

    $ mkdir bin && cd bin && cmake -D WITH_CUDA=OFF [-D WITH_QT=ON] ..
    $ make


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

Set-up on OS X
==============

This has only been tested on OS X El Capitan.

- OS X El Capitan should already have Python 2.7 installed so you
  should not have to install that.

- Install Xcode 5 or higher. Seems to be required for intalling
  Qt. Also, you must open Xcode and accept its license in order to
  install some additional components like the C++ compiler etc.

- Install Homebrew (visit brew.sh for instructions). This will help
  install some other dependencies easily.

- Install cmake, boost, libtiff, glog (this should automatically
  also install gflags), ceres-solver and gperftools:

.. code-block :: bash

    $ brew install cmake
    $ brew install boost
    $ brew install boost-python
    $ brew install libtiff
    $ brew install glog
    $ brew install ceres-solver (or brew install
    homebrew/science/ceres-solver)
    $ brew install gperftools

- Install CUDA Toolkit for OS X from developer.nvidia.com. As
  mentioned earlier, you need to install this right now regardless if
  whether you have a CUDA capable GPU on your machine since the
  toolkit libraries are currently a required set of dependency. Again,
  feel free to skip installing the driver and samples, which is what I
  did while testing this on a 13 inch Macbook Pro.  
 
- Install Qt 4 or higher (visit qt.io for instructions)

- Install OpenCV (2.4.11 built ok) from source. Build this using the
  following cmake configuration:

.. code-block :: bash

    $ cd opencv-2.4.11
    $ mkdir build
    $ cd build
    $ cmake -D WITH_PYTHON=ON -D WITH_CUDA=ON -D
    CUDA_GENERATION=Kepler ..
    $ make

- Build OpenFV:

.. code-block :: bash

    $ cd openfv
    $ ./configure
    $ cd bin
    $ cmake [-D OpenCV_DIR=<path to where OpenCV is built> -D
    EIGEN_INC_DIR=<path to where Eigen include files lie> -D
    PYTHON_LIBRARIES=<path to where libpython2.7.dylib lies> -D
    PYTHON_INCLUDE_DIR=<path to where pyconfig.h and Python.h lie>] ..
    $ make
