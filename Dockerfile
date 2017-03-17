# Copyright (c) OpenFV
# Distributed under the GNU General Public License 
# as published by the Free Software Foundation

FROM jupyter/scipy-notebook

MAINTAINER Abhishek Bajpayee <ab9@mit.edu>

USER root

# Installing OpenFV dependencies
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
	libtool

# Installing Glog
RUN git clone https://github.com/google/glog.git && \
	autoreconf --force --install glog && \
	cd glog && ./configure && make && make install

# Installing Ceres Solver
RUN wget http://ceres-solver.org/ceres-solver-1.11.0.tar.gz && \
	tar -xvzf ceres-solver-1.11.0.tar.gz && \
	cd ceres-solver-1.11.0 && \
	mkdir build && cd build && \
	cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC .. && make && make install

# Add CUDA
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 7.0
LABEL com.nvidia.cuda.version="7.0"

ENV CUDA_PKG_VERSION 7-0=7.0-28
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-$CUDA_PKG_VERSION \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-$CUDA_VERSION /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install OpenCV 3
RUN wget https://github.com/Itseez/opencv/archive/3.0.0.zip
RUN unzip 3.0.0.zip

# RUN cd opencv-3.0.0 && mkdir build && cd build && \
#	cmake -D CUDA_GENERATION=Kepler -D WITH_QT=OFF ..
# took the sudo off "sudo make install" Azuh
# RUN make && make install

# Remove files

# USER $NB_USER

# Install OpenFV
# RUN cd ../.. && git clone https://github.com/abhishekbajpayee/openfv.git && \

RUN cd opencv-3.0.0 && mkdir build && cd build && \
        cmake -D CUDA_GENERATION=Kepler .. && make -j7 && make install

USER $NB_USER

RUN git clone https://github.com/abhishekbajpayee/openfv.git && \
	cd openfv && ./configure && cd bin && \
	cmake -D BUILD_PYTHON=ON -D WITH_CUDA=ON -DCMAKE_CXX_FLAGS=-isystem /usr/local/include/opencv2 .. 
