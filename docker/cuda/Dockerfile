FROM openfv-scipy-notebook

MAINTAINER Abhishek Bajpayee <ab9@mit.edu>

USER root

RUN apt-get update -qq
RUN apt-get install -yq --no-install-recommends qt5-default \
        cmake \
        build-essential \
        libboost-all-dev \
        libgoogle-perftools-dev \
        google-perftools \
        libeigen3-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
        libtiff5-dev \
        autoconf \
        automake \
        libtool \
        git \
        wget \
        unzip \
        ca-certificates

# Installing Glog
RUN git clone https://github.com/google/glog.git && \
        autoreconf --force --install glog && \
        cd glog && ./configure && make && make install

# Installing Gflags
RUN git clone https://github.com/gflags/gflags.git && \
        cd gflags && mkdir bin && cd bin && \
        cmake .. && make && make install

# Installing Ceres Solver
RUN wget http://ceres-solver.org/ceres-solver-1.11.0.tar.gz && \
        tar -xvzf ceres-solver-1.11.0.tar.gz && \
        cd ceres-solver-1.11.0 && \
        mkdir build && cd build && \
        cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC .. && make && make install

# Install OpenCV
RUN wget https://github.com/Itseez/opencv/archive/2.4.10.zip
RUN unzip 2.4.10.zip
RUN cd opencv-2.4.10 && mkdir build && cd build && \
        cmake -D CUDA_GENERATION=Kepler .. && make -j7 && make install

# Install openfv
RUN git clone https://github.com/abhishekbajpayee/openfv.git && \
        cd openfv && ./configure && cd bin && \
        cmake -D BUILD_PYTHON=ON -D WITH_CUDA=ON .. && \
        make && make install

RUN rm -rf *.zip ceres-solver* gflags glog opencv* openfv

# Download sample data
RUN git clone https://github.com/abhishekbajpayee/openfv-sample-data.git

# Download and test sample
RUN git clone https://github.com/abhishekbajpayee/openfv-samples.git # && cd openfv-samples && \
        # mkdir bin && cd bin && cmake .. && make -j4

USER $NB_USER
