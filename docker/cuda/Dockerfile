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

FROM openfv/openfv-cuda-base

# Install openfv
RUN git clone https://github.com/abhishekbajpayee/openfv.git && \
        cd openfv && ./configure && cd bin && \
        cmake -D BUILD_PYTHON=ON -D PYTHON_EXEC=/opt/conda/envs/python2/bin/python -D WITH_CUDA=ON \
        -D NUMPY_INC_DIR=/opt/conda/envs/python2/lib/python2.7/site-packages/numpy/core/include ..  && \
        make && make install

# RUN rm -rf *.zip ceres-solver* gflags glog opencv* openfv
RUN rm -rf *.zip gflags glog opencv* openfv

# Download sample data
RUN git clone https://github.com/abhishekbajpayee/openfv-sample-data.git

# Download and test sample
RUN git clone https://github.com/abhishekbajpayee/openfv-samples.git && cd openfv-samples && \
        mkdir bin && cd bin && cmake .. && make

# Add local files as late as possible to avoid cache busting
COPY start.sh /usr/local/bin/
COPY start-notebook.sh /usr/local/bin/
COPY start-singleuser.sh /usr/local/bin/
COPY jupyter_notebook_config.py /home/$NB_USER/.jupyter/

# Configure ipython kernel to use matplotlib inline backend by default
RUN mkdir -p $HOME/.ipython/profile_default/startup
COPY mplimporthook.py $HOME/.ipython/profile_default/startup/

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER
