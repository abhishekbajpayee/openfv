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

FROM bpaez1/openfv:cudabase

# Install openfv
RUN git clone https://github.com/bpaez/openfv.git && \
        cd openfv && \
        ./configure && cd bin && \
	#cmake -D BUILD_PYTHON=ON -D PYTHON_EXEC=/opt/conda/envs/python2/bin/python -D WITH_CUDA=ON \
	cmake -D WITH_CUDA=ON .. && \
        #-D NUMPY_INC_DIR=/opt/conda/envs/python2/lib/python2.7/site-packages/numpy/core/include ..  && \
    	make && make install

RUN rm -rf *.zip gflags glog opencv* 

# Download sample repo (note: downloading it bc current repo is private)
RUN git clone https://gitlab.com/Bpaez/FILMopenfv-samples.git && \
    cd FILMopenfv-samples && mkdir bin && cd bin && \
    cmake .. && make

# Add aliases so users can depend on a consistent format
RUN echo 'alias refCalib="python3 refCalib.py -c "' >> ~/.bashrc && \
    echo 'alias pinCalib="python3 pinCalib.py -c "' >> ~/.bashrc && \
    echo 'alias refocus="./refocus --config_file "' >> ~/.bashrc 

# Add local files as late as possible to avoid cache busting
#COPY cuda/start.sh /usr/local/bin/
#COPY cuda/start-notebook.sh /usr/local/bin/
#COPY cuda/start-singleuser.sh /usr/local/bin/
#COPY cuda/jupyter_notebook_config.py /home/$NB_USER/.jupyter/

# Configure ipython kernel to use matplotlib inline backend by default
#RUN mkdir -p $HOME/.ipython/profile_default/startup
#COPY cuda/mplimporthook.py $HOME/.ipython/profile_default/startup/

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER
