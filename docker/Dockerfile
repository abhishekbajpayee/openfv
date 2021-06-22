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

# Install boost
RUN apt-get update -qq && apt-get install libboost-all-dev -y

# Install openfv
RUN git clone https://github.com/bpaez/openfv.git && \
	#apt-get update -qq && apt-get install libboost-all-dev -y && \
        cd openfv && \
        ./configure && cd bin && \
	cmake -D WITH_CUDA=ON .. && \
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

USER $NB_USER
