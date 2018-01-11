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

FROM ubuntu:14.04

USER root

# Install all OS dependencies for fully functioning notebook server,
# libav-tools for matplotlib anim,
# and OpenFV dependencies
RUN apt-get update -qq && apt-get -yq dist-upgrade && \
    apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    git \
    vim \
    jed \
    emacs \
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
    inkscape \
    libav-tools \
    libx11-xcb-dev \
    mesa-common-dev \
    libglu1-mesa-dev \
    qt5-default \
    libqt5opengl5 \
    x11-apps \
    cmake \
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

ADD cuda/fix-permissions /usr/local/bin/fix-permissions

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
ENV MINICONDA_VERSION 4.3.30
RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "0b80a152332a4ce5250f3c09589c7a81 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda update --all --quiet --yes
RUN conda clean -tipsy && \
    fix-permissions $CONDA_DIR

# Install Jupyter Notebook and Hub
RUN conda install -c conda-forge --quiet --yes \
    'notebook=5.2.*' \
    'jupyterhub=0.8.*' \
    'jupyterlab=0.30.*' && \
    conda clean -tipsy && \
    jupyter labextension install @jupyterlab/hub-extension@^0.7.0 && \
    npm cache clean && \
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

# Not installing Python 3 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
# RUN conda install --quiet --yes \
#     'nomkl' \
#     'ipywidgets=6.0*' \
#     'pandas=0.19*' \
#     'numexpr=2.6*' \
#     'matplotlib=2.0*' \
#     'scipy=0.19*' \
#     'seaborn=0.7*' \
#     'scikit-learn=0.18*' \
#     'scikit-image=0.12*' \
#     'sympy=1.0*' \
#     'cython=0.25*' \
#     'patsy=0.4*' \
#     'statsmodels=0.8*' \
#     'cloudpickle=0.2*' \
#     'dill=0.2*' \
#     'numba=0.31*' \
#     'bokeh=0.12*' \
#     'sqlalchemy=1.1*' \
#     'hdf5=1.8.17' \
#     'h5py=2.6*' \
#     'vincent=0.4.*' \
#     'beautifulsoup4=4.5.*' \
#     'xlrd'  && \
#     conda remove --quiet --yes --force qt pyqt && \
#     conda clean -tipsy

# Activate ipywidgets extension in the environment that runs the notebook server
# RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Install Python 2 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
RUN conda create --quiet --yes -p $CONDA_DIR/envs/python2 python=2.7 \
    'nomkl' \
    'ipython=5.3*' \
    'ipywidgets=6.0*' \
    'pandas=0.19*' \
    'numexpr=2.6*' \
    'matplotlib=2.0*' \
    'scipy=0.19*' \
    'seaborn=0.7*' \
    'scikit-learn=0.18*' \
    'scikit-image=0.12*' \
    'sympy=1.0*' \
    'cython=0.25*' \
    'patsy=0.4*' \
    'statsmodels=0.8*' \
    'cloudpickle=0.2*' \
    'dill=0.2*' \
    'numba=0.31*' \
    'bokeh=0.12*' \
    'hdf5=1.8.17' \
    'h5py=2.6*' \
    'sqlalchemy=1.1*' \
    'pyzmq' \
    'vincent=0.4.*' \
    'beautifulsoup4=4.5.*' \
    'xlrd' && \
    conda remove -n python2 --quiet --yes --force qt pyqt && \
    conda clean -tipsy
# Add shortcuts to distinguish pip for python2 and python3 envs
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
# RUN wget http://ceres-solver.org/ceres-solver-1.11.0.tar.gz && \
# 	tar -xvzf ceres-solver-1.11.0.tar.gz && \
# 	cd ceres-solver-1.11.0 && \
# 	mkdir build && cd build && \
# 	cmake -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC .. && make && make install

# Install OpenCV
RUN wget https://github.com/Itseez/opencv/archive/2.4.10.zip
RUN unzip 2.4.10.zip
RUN cd opencv-2.4.10 && mkdir build && cd build && \
        cmake -D WITH_QT=ON .. && make -j7 && make install    

# Install openfv
RUN git clone https://github.com/abhishekbajpayee/openfv.git && \
	cd openfv && ./configure && cd bin && \
	cmake -D BUILD_PYTHON=ON -D PYTHON_EXEC=/opt/conda/envs/python2/bin/python -D WITH_CUDA=OFF .. && \
    make && make install

# RUN rm -rf *.zip ceres-solver* gflags glog opencv* openfv
RUN rm -rf *.zip gflags glog opencv* openfv

# Download sample data
RUN git clone https://github.com/abhishekbajpayee/openfv-sample-data.git

# Download and test sample
RUN git clone https://github.com/abhishekbajpayee/openfv-samples.git # && cd openfv-samples && \
     	mkdir bin && cd bin && cmake .. && make

# Add local files as late as possible to avoid cache busting
COPY cuda/start.sh /usr/local/bin/
COPY cuda/start-notebook.sh /usr/local/bin/
COPY cuda/start-singleuser.sh /usr/local/bin/
COPY cuda/jupyter_notebook_config.py /home/$NB_USER/.jupyter/

# Configure ipython kernel to use matplotlib inline backend by default
RUN mkdir -p $HOME/.ipython/profile_default/startup
COPY cuda/mplimporthook.py $HOME/.ipython/profile_default/startup/

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER
