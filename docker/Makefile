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

# Makefile to build OpenFV docker images

# TODO: doesn't seem to be working properly
remove-untagged-images:
	@-(docker rmi $$(docker images | grep "^<none>" | awk '{print $3}')) > /dev/null

build-image:
	@docker build -t openfv/openfv .

build-cuda-image:
	@-(cd cuda && nvidia-docker build -t openfv/openfv-cuda .)

pull-images:
	@docker pull openfv/openfv
	@docker pull openfv/openfv-cuda

push-images:
	@docker push openfv/openfv
	@docker push openfv/openfv-cuda

images: build-image build-cuda-image
