#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#                           License Agreement
#                For Open Source Flow Visualization Library
#
# Copyright 2013-2017 Abhishek Bajpayee
#
# This file is part of OpenFV.
#
# OpenFV is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License version 2 as published by the Free Software Foundation.
#
# OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License version 2 for more details.
#
# You should have received a copy of the GNU General Public License version 2 along with
# OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

FROM ubuntu:16.04

USER root

# Install all OS dependencies for fully functioning notebook server,
# libav-tools for matplotlib anim,
# and OpenFV dependencies
RUN apt-get update -qq && apt-get -yq dist-upgrade && \
    apt-get autoclean && apt-get clean && \
    apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    git \
    vim \
    jed \
    #emacs \
    build-essential \
    python-dev \
    unzip \
    libsm6 \
    pandoc \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    texlive-xetex \
    libxrender1 \
    #inkscape \
    libav-tools \
    libx11-xcb-dev \
    mesa-common-dev \
    libglu1-mesa-dev \
    qt5-default \
    libqt5opengl5 \
    x11-apps \
    #cmake \
    build-essential \
    libboost-all-dev \
    libgoogle-perftools-dev \
    google-perftools \
    libeigen3-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libtiff5-dev \
    libyaml-cpp-dev \
    autoconf \
    automake \
    libtool && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CMAKE
RUN apt-get update && apt-get upgrade && apt remove --purge --auto-remove cmake && apt-get install -y libssl-dev
RUN version=3.16 && build=5 && mkdir ~/temp && \
    cd ~/temp && wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && \
    cd cmake-$version.$build/ && \
    ./bootstrap && \
    make -j2 && sudo make install

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.10.0/tini && \
    echo "1361527f39190a7338a0b434bd8c88ff7233ce7b9a4876f3315c22fce7eca1b0 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
    PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

# Berlin: Testing, changed directory of Dockerfile
#ADD cuda/fix-permissions /usr/local/bin/fix-permissions
ADD fix-permissions /usr/local/bin/fix-permissions
#ADD fix-permissions .
#COPY fix-permissions .

# Create jovyan user with UID=1000 and in the 'users' group
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

USER $NB_USER

# Setup jovyan home directory
RUN mkdir /home/$NB_USER/work && \
    mkdir /home/$NB_USER/.jupyter && \
    echo "cacert=/etc/ssl/certs/ca-certificates.crt" > /home/$NB_USER/.curlrc \
    fix-permissions $HOME

# Allow python wrappers to find shared
# RUN echo "export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/openfv/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc

# Install conda as jovyan and check the md5 sum provided on the download site
# Berlin: Changed some urls to fit with updated syntax
#ENV MINICONDA_VERSION 4.3.30, replaced with "latest"
RUN cd /tmp && \
    #wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    echo "87e77f097f6ebb5127c77662dfc3165e *Miniconda3-latest-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    #rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda update --all --quiet --yes
RUN conda clean -tipsy && \
    fix-permissions $CONDA_DIR

# Install Jupyter Notebook and Hub
RUN conda install -c conda-forge --quiet --yes \
    #'notebook=5.2.*' \
    'notebook' \
    #'jupyterhub=0.8.*' \
    'jupyterhub' \
    #'jupyterlab=0.30.*' && \
    'jupyterlab' && \
    conda clean -tipsy && \
    jupyter labextension install @jupyterlab/hub-extension && \
    #npm cache clean && \
    npm cache verify && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    fix-permissions $CONDA_DIR

USER root

EXPOSE 8888
WORKDIR /home/$NB_USER/work

# Configure container startup
ENTRYPOINT ["tini", "--"]
CMD ["cuda/start-notebook.sh"]

RUN chown -R $NB_USER:users /home/$NB_USER/.jupyter

USER $NB_USER

# Activate ipywidgets extension in the environment that runs the notebook server
# RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Install Python packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
# Berlin: removed the variables specifying build version
RUN conda update conda && \
    conda create --quiet --yes -p $CONDA_DIR/envs/python2 python=2.7 \
    'nomkl' \
    'ipython' \
    'ipywidgets' \
    'pandas' \
    'numexpr' \
    'matplotlib' \
    'scipy' \
    'seaborn' \
    'scikit-learn' \
    'scikit-image' \
    'sympy' \
    'cython' \
    'patsy' \
    'statsmodels' \
    'cloudpickle' \
    'dill' \
    'numba' \
    'bokeh' \
    'hdf5' \
    'h5py' \
    'sqlalchemy' \
    'pyzmq' \
    'vincent' \
    'beautifulsoup4' \
    'xlrd' && \
    conda remove -n python2 --quiet --yes --force qt pyqt && \
    conda clean -tipsy
# Add shortcuts to distinguish pip for python2 and python3 envs
#RUN ln -s $CONDA_DIR/envs/python2/bin/pip $CONDA_DIR/bin/pip2 && \
RUN ln -s $CONDA_DIR/envs/python2/bin/pip $CONDA_DIR/bin/pip2 && \
    ln -s $CONDA_DIR/bin/pip $CONDA_DIR/bin/pip3

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg $CONDA_DIR/envs/python2/bin/python -c "import matplotlib.pyplot"

USER root

# Install Python 2 kernel spec globally to avoid permission problems when NB_UID
# switching at runtime and to allow the notebook server running out of the root
# environment to find it. Also, activate the python2 environment upon kernel
# launch.
RUN pip install kernda --no-cache && \
    $CONDA_DIR/envs/python2/bin/python -m ipykernel install && \
    kernda -o -y /usr/local/share/jupyter/kernels/python2/kernel.json && \
    pip uninstall kernda -y

# Installing glog
RUN git clone https://github.com/google/glog.git && \
	autoreconf --force --install glog && \
	cd glog && ./configure && make && make install

# Installing gflags
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
# Berlin: OpenCv 3.1.0 does have the cuda modules along with regular modules!
#RUN wget https://github.com/opencv/opencv/archive/3.4.10.zip && \
	#unzip 3.4.10.zip && rm -r 3.4.10.zip
#RUN CUDA_BIN_PATH=/usr/local/cuda
RUN git clone https://github.com/opencv/opencv.git && \
	cd opencv && \
	#git checkout 3.3.1 && \
	git checkout 2.4.10 && \
	cd ..
#RUN git clone https://github.com/opencv/opencv_contrib.git && \
	#cd opencv_contrib && \
	#git checkout 3.3.1 && \
	#git checkout 2.4.10 && \
	#cd ..
RUN cd opencv && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_TBB=ON \
	-D WITH_V4L=ON \
	-D WITH_QT=ON \
	-D WITH_OPENGL=ON \
	-D WITH_CUDA=ON \
	#-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	.. && make -j3 && make install
#RUN cd opencv-3.4.10 && mkdir build && cd build && \
	#cmake -D WITH_QT=ON \
	#-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 \
	#-D WITH_CUDA=ON \
	#-D BUILD_opencv_cudaarithm=ON \
	#-D PYTHON_DEFAULT_EXECUTABLE=$CONDA_DIR/envs/python2/bin/python \
	#-D PYTHON_INCLUDE_DIRS=$CONDA_DIR/envs/python2/include/python \
	#-D PYTHON_LIBRARY=$CONDA_DIR/envs/python2/lib/libpython2.7.so \
	#-D PYTHON2_NUMPY_INCLUDE_DIRS=$CONDA_DIR/envs/python2/lib/python2.7/site-packages/numpy/core/include \
	#-D PYTHON_EXECUTABLE=$CONDA_DIR/envs/python2/bin/python2 .. && \
	#.. && \
	#make -j3 && make install
