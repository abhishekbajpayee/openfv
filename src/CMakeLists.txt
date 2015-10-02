# IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
# 
# By downloading, copying, installing or using the software you agree to this license.
# If you do not agree to this license, do not download, install,
# copy or use the software.
# 
#                            License Agreement
#                 For Open Source Flow Visualization Library
# 
# Copyright 2013-2015 Abhishek Bajpayee
# 
# This file is part of openFV.
# 
# openFV is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 
# 3 of the License, or (at your option) any later version.
# 
# openFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with openFV. 
# If not, see http://www.gnu.org/licenses/.

add_library (openfv 
		   SHARED
		   ${PROJECT_SOURCE_DIR}/src/modules/optimization.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp
		   ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/visualize.cpp
		   ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp
                   #${PROJECT_SOURCE_DIR}/src/modules/batchProc.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/parse_settings.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/parse_safe_settings.cpp
	  	   ${PROJECT_SOURCE_DIR}/src/modules/featureDetection.cpp
	  	   ${PROJECT_SOURCE_DIR}/src/modules/safeRefocusing.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/piv.cpp)

target_link_libraries(openfv ${LIBS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)
cuda_add_library (cuda_openfv SHARED ${PROJECT_SOURCE_DIR}/src/modules/cuda_lib.cu)

if(BUILD_PYTHON)
  add_library(rendering SHARED ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp)
  set_target_properties(rendering PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(rendering ${LIBS})

  add_library(refocusing SHARED ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp)
  set_target_properties(refocusing PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(refocusing ${LIBS})

  add_library(tools SHARED ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp)
  set_target_properties(tools PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(tools ${LIBS})

  add_library(tracking SHARED ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
  set_target_properties(tracking PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(tracking ${LIBS})

  add_library(calibration SHARED ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp)
  set_target_properties(calibration PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(calibration ${LIBS})

# add_library(localization SHARED ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp)
# set_target_properties(localization PROPERTIES ${PYTHON_PROPS} PREFIX "")
# target_link_libraries(localization ${LIBS})
 
  install(TARGETS rendering refocusing tools tracking calibration
          DESTINATION /usr/local/lib)
endif()

install(TARGETS cuda_openfv openfv DESTINATION /usr/local/lib)